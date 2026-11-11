"""k-nearest-neighbour graphs for UMAP built with a Faiss IVF-Flat index.

The coarse quantizer of the index is trained by e-means
(``faiss.contrib.e_means.NumpyEmeans``) or by Faiss' Lloyd k-means. Every
graph returned here is in the format ``umap.UMAP(precomputed_knn=...)``
expects: exactly ``n_neighbors`` C-contiguous columns, the point itself in
column 0 at distance 0, rows sorted by increasing true distance (Euclidean
norm, or ``1 - cos`` for cosine), float32 distances and int32 indices.
"""

from __future__ import annotations

import numbers
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator, NamedTuple

import faiss
import numpy as np
import scipy.sparse
from faiss.contrib.e_means import NumpyEmeans

METRICS = ("euclidean", "cosine")
MIN_POINTS_PER_LIST = 39  # Faiss' own lower bound for k-means training
_BATCH = 262144


class KNNGraph(NamedTuple):
    """Result of :func:`ivf_knn`."""

    indices: np.ndarray
    distances: np.ndarray
    index: "FaissKNNIndex"
    info: dict[str, Any]


def _check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(
            f"metric={metric!r} is not supported by eumap (only {METRICS}); "
            "use umap.UMAP directly for other metrics."
        )


def _check_k(n: int, n_neighbors: int) -> None:
    if not 2 <= n_neighbors <= n:
        raise ValueError(f"n_neighbors must be in [2, n_samples={n}], got {n_neighbors}.")


def _faiss_metric(metric: str) -> int:
    return faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2


def _index_dtype(n: int) -> type:
    return np.int32 if n < 2**31 else np.int64


def available_cpus() -> int:
    """Number of CPUs in the process' affinity mask (``os.cpu_count()`` where there is none)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@contextmanager
def faiss_threads(n_threads: int | None) -> Iterator[int]:
    """Set Faiss' OpenMP thread count inside the block and restore it afterwards.

    ``None`` means every CPU in the process' affinity mask.
    """
    if n_threads is None:
        n_threads = available_cpus()
    if int(n_threads) < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}.")
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(int(n_threads))
    try:
        yield int(n_threads)
    finally:
        faiss.omp_set_num_threads(previous)


def prepare(X: Any, metric: str) -> np.ndarray:
    """Return ``X`` as a float32 C-contiguous copy ready for Faiss.

    For ``"euclidean"`` the copy is centred (the column means, computed in
    float64, are subtracted). Translation does not change Euclidean distances,
    but Faiss computes them as ``|x|^2 + |y|^2 - 2<x, y>`` in float32, which
    loses most of its precision when the data share a large common offset.
    For ``"cosine"`` the copy is L2-normalised. The caller's array is never
    modified.

    Raises
    ------
    ValueError
        Unsupported metric, input that is not 2-D or empty, or non-finite values.
    TypeError
        Sparse input.
    """
    return _prepare(X, metric)[0]


def _prepare(X: Any, metric: str, shift: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """:func:`prepare`, also returning the subtracted column means (None for cosine).

    With ``shift`` given (Euclidean only), that vector is subtracted instead of
    the column means of ``X``: this is how new query points are moved into the
    frame of an index.
    """
    _check_metric(metric)
    if scipy.sparse.issparse(X):
        raise TypeError("eumap needs dense input; use umap.UMAP for sparse data.")
    Xp = np.array(X, dtype=np.float32, order="C", copy=True)
    if Xp.ndim != 2 or Xp.shape[0] == 0 or Xp.shape[1] == 0:
        raise ValueError(f"X must be a non-empty 2-D array, got shape {Xp.shape}.")
    if not np.isfinite(Xp).all():
        raise ValueError("X contains NaN or infinite values.")
    if metric == "cosine":
        faiss.normalize_L2(Xp)
        return Xp, None
    if shift is None:
        shift = Xp.mean(axis=0, dtype=np.float64).astype(np.float32)
    elif shift.shape != (Xp.shape[1],):
        raise ValueError(f"X has {Xp.shape[1]} columns, the index has {shift.shape[0]}.")
    Xp -= shift
    return Xp, shift


def default_nlist(n: int) -> int:
    """About ``4 * sqrt(n)`` rounded to a power of two, capped at ``n // 39``, at least 1."""
    return int(max(1, min(2 ** round(np.log2(4 * np.sqrt(n))), n // MIN_POINTS_PER_LIST)))


def train_centroids(
    Xp: np.ndarray,
    nlist: int,
    trainer: str = "emeans",
    *,
    spherical: bool = False,
    seed: int = 1234,
    **trainer_kwds: Any,
) -> np.ndarray:
    """Train ``nlist`` coarse centroids on prepared data.

    Parameters
    ----------
    Xp : ndarray of shape (n, d)
        Output of :func:`prepare`.
    nlist : int
        Number of centroids; at most ``n // 39``.
    trainer : {"emeans", "lloyd"}
        ``"emeans"`` uses ``NumpyEmeans`` (default 5 epochs); ``"lloyd"`` uses
        ``faiss.Kmeans`` with the settings of Faiss' own IVF training
        (``niter=10, max_points_per_centroid=256``).
    spherical : bool
        Normalised centroids, for cosine.
    seed : int
        Seed of the trainer (taken modulo ``2**31`` for Lloyd, whose seed is a C int).
    **trainer_kwds
        Passed to the trainer, overriding the defaults above.

    Returns
    -------
    ndarray of shape (nlist, d), float32
    """
    n, d = Xp.shape
    _check_nlist(n, nlist)
    if trainer not in ("emeans", "lloyd"):
        raise ValueError(f"trainer must be 'emeans' or 'lloyd', got {trainer!r}.")
    if trainer == "emeans":
        centroids = NumpyEmeans(d, nlist, seed=seed, spherical=spherical, **trainer_kwds).train(Xp)[0]
    else:
        kwds = {"niter": 10, "max_points_per_centroid": 256, **trainer_kwds}
        # Faiss' seed is a C int; reduce wider seeds (UMAP accepts up to 2**32 - 1).
        km = faiss.Kmeans(d, nlist, seed=int(seed) % 2**31, spherical=spherical, **kwds)
        km.train(Xp)
        centroids = km.centroids
    return np.ascontiguousarray(centroids, dtype=np.float32)


def _check_nlist(n: int, nlist: Any) -> None:
    if n < MIN_POINTS_PER_LIST:
        raise ValueError(
            f"an IVF index needs at least {MIN_POINTS_PER_LIST} samples, got {n}; "
            "use exact_knn (or EUMAP's default exact_below) for data this small."
        )
    if not isinstance(nlist, numbers.Integral) or not 1 <= nlist <= n // MIN_POINTS_PER_LIST:
        raise ValueError(
            f"nlist={nlist} needs at least {MIN_POINTS_PER_LIST} points per list, "
            f"i.e. 1 <= nlist <= n // {MIN_POINTS_PER_LIST} = {n // MIN_POINTS_PER_LIST}."
        )


def build_ivf_index(Xp: np.ndarray, centroids: np.ndarray, metric: str) -> faiss.IndexIVFFlat:
    """Build an ``IndexIVFFlat`` around the given centroids and add every row of ``Xp``."""
    _check_metric(metric)
    nlist, d = centroids.shape
    if d != Xp.shape[1]:
        raise ValueError(f"centroids have dimension {d}, data has {Xp.shape[1]}.")
    quantizer = faiss.IndexFlatIP(d) if metric == "cosine" else faiss.IndexFlatL2(d)
    quantizer.add(np.ascontiguousarray(centroids, dtype=np.float32))
    index = faiss.IndexIVFFlat(quantizer, d, nlist, _faiss_metric(metric))
    # Faiss would otherwise silently re-run k-means on add/train.
    if quantizer.ntotal != nlist or not index.is_trained:
        raise RuntimeError("the IVF quantizer did not take the given centroids.")
    index.add(Xp)
    return index


def _search(index: faiss.Index, Xq: np.ndarray, k: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw Faiss search in query batches: (squared L2 or inner product, int64 ids)."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    D = np.empty((Xq.shape[0], k), dtype=np.float32)
    I = np.empty((Xq.shape[0], k), dtype=np.int64)
    for s in range(0, Xq.shape[0], batch_size):
        D[s : s + batch_size], I[s : s + batch_size] = index.search(Xq[s : s + batch_size], k)
    return D, I


def _brute(Xq: np.ndarray, Xp: np.ndarray, k: int, metric: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact search of ``Xq`` against ``Xp`` in Faiss' scale (squared L2 or inner product).

    Faiss' brute force uses the BLAS form ``|x|^2 + |y|^2 - 2<x, y>``, whose
    rounding residue gives exact duplicates a non-zero distance. The scores of
    the returned pairs are therefore recomputed directly and the rows re-sorted.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    D = np.empty((Xq.shape[0], k), dtype=np.float32)
    I = np.empty((Xq.shape[0], k), dtype=np.int64)
    for s in range(0, Xq.shape[0], batch_size):
        q = Xq[s : s + batch_size]
        _, I[s : s + batch_size] = faiss.knn(q, Xp, k, metric=_faiss_metric(metric))
        D[s : s + batch_size] = _rescore(q, Xp, I[s : s + batch_size], metric)
    order = np.argsort(-D if metric == "cosine" else D, axis=1, kind="stable")
    return np.take_along_axis(D, order, axis=1), np.take_along_axis(I, order, axis=1)


def _rescore(Xq: np.ndarray, Xp: np.ndarray, I: np.ndarray, metric: str) -> np.ndarray:
    """Squared L2 (or inner product) between each row of ``Xq`` and its ids ``I``, computed directly.

    For cosine (unit vectors) the inner product is taken as ``1 - |x - y|^2 / 2``,
    which is exact for duplicates and keeps the precision of small distances.
    """
    out = np.empty(I.shape, dtype=np.float32)
    step = max(1, 2**22 // (I.shape[1] * Xq.shape[1]))  # about 16 MB of gathered vectors per step
    for s in range(0, Xq.shape[0], step):
        diff = Xp[I[s : s + step]] - Xq[s : s + step, None, :]
        out[s : s + step] = np.einsum("ijd,ijd->ij", diff, diff)
    if metric == "cosine":
        out = 1.0 - 0.5 * out
    return out


def _true_distances(D: np.ndarray, I: np.ndarray, metric: str) -> np.ndarray:
    """Faiss scores to true distances, with ``inf`` for missing (-1) neighbours."""
    D = np.maximum(1.0 - D, 0.0) if metric == "cosine" else np.sqrt(np.maximum(D, 0.0))
    D[I < 0] = np.inf
    return D.astype(np.float32, copy=False)


def _self_first(D: np.ndarray, I: np.ndarray, self_ids: np.ndarray) -> None:
    """Put each query's own id in column 0 at distance 0, in place.

    Needed for duplicate rows (ties go to the lowest id) and for float rounding.
    """
    k = I.shape[1]
    for r in np.flatnonzero(I[:, 0] != self_ids):
        keep = I[r] != self_ids[r]
        I[r] = np.concatenate(([self_ids[r]], I[r][keep]))[:k]
        D[r] = np.concatenate(([0.0], D[r][keep]))[:k]
    D[:, 0] = 0.0


def _graph(D: np.ndarray, I: np.ndarray, metric: str, self_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = _true_distances(D, I, metric)
    _self_first(D, I, self_ids)
    return I, D


def _check_nprobe(nprobe: Any, nlist: int) -> None:
    if isinstance(nprobe, bool) or not isinstance(nprobe, numbers.Integral) or not 1 <= nprobe <= nlist:
        raise ValueError(f"nprobe must be 'auto' or an int in [1, nlist={nlist}], got {nprobe!r}.")


def _check_target_recall(target_recall: Any) -> None:
    if not isinstance(target_recall, numbers.Real) or not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall!r}.")


def _search_self(
    index: faiss.IndexIVF, Xp: np.ndarray, n_neighbors: int, nprobe: int, metric: str, batch_size: int, repair: bool
) -> tuple[np.ndarray, np.ndarray, int]:
    n = Xp.shape[0]
    _check_k(n, n_neighbors)
    _check_nprobe(nprobe, index.nlist)
    index.nprobe = int(nprobe)
    ids = np.arange(n)
    I, D = _graph(*_search(index, Xp, n_neighbors, batch_size), metric, ids)
    rows = np.flatnonzero((I < 0).any(axis=1))
    if repair and rows.size:
        I[rows], D[rows] = _graph(*_brute(Xp[rows], Xp, n_neighbors, metric, batch_size), metric, rows)
    return np.ascontiguousarray(I.astype(_index_dtype(n))), np.ascontiguousarray(D), int(rows.size)


def search_self(
    index: faiss.IndexIVF,
    Xp: np.ndarray,
    n_neighbors: int,
    nprobe: int,
    metric: str,
    *,
    batch_size: int = _BATCH,
    repair: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """kNN graph of the indexed points against themselves, in UMAP's format.

    Sets ``index.nprobe``. With ``repair=True``, rows that came back with
    missing neighbours are re-queried exactly so the graph is complete; with
    ``repair=False`` missing slots stay ``-1`` / ``inf``.

    Returns
    -------
    indices : ndarray of shape (n, n_neighbors), int32 (int64 for n >= 2**31)
    distances : ndarray of shape (n, n_neighbors), float32
    """
    I, D, _ = _search_self(index, Xp, n_neighbors, nprobe, metric, batch_size, repair)
    return I, D


def exact_knn(
    X: Any, n_neighbors: int, metric: str = "euclidean", *, batch_size: int = _BATCH
) -> tuple[np.ndarray, np.ndarray]:
    """Exact kNN graph by Faiss brute force, in the same format as :func:`search_self`."""
    return _exact(prepare(X, metric), n_neighbors, metric, batch_size)


def _exact(Xp: np.ndarray, n_neighbors: int, metric: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = Xp.shape[0]
    _check_k(n, n_neighbors)
    I, D = _graph(*_brute(Xp, Xp, n_neighbors, metric, batch_size), metric, np.arange(n))
    return np.ascontiguousarray(I.astype(_index_dtype(n))), D


def knn_recall(indices: np.ndarray, true_indices: np.ndarray) -> float:
    """Mean fraction of the true non-self neighbours found, per row.

    Column 0 (the point itself) is ignored, so the result is the overlap of
    columns ``1:`` divided by ``n_neighbors - 1``. Missing (-1) entries never match.
    """
    if indices.shape != true_indices.shape or indices.ndim != 2 or indices.shape[1] < 2:
        raise ValueError(f"shape mismatch or too few columns: {indices.shape} vs {true_indices.shape}.")
    hits = 0
    step = max(1, 2**24 // indices.shape[1] ** 2)  # bounds the (rows, k-1, k-1) comparison to about 16 MB
    for s in range(0, indices.shape[0], step):
        a = indices[s : s + step, 1:, None]
        b = true_indices[s : s + step, None, 1:]
        hits += int(((a == b) & (a >= 0)).any(axis=2).sum())
    return hits / (indices.shape[0] * (indices.shape[1] - 1))


def tune_nprobe(
    index: faiss.IndexIVF,
    Xp: np.ndarray,
    n_neighbors: int,
    metric: str,
    *,
    target_recall: float = 0.95,
    n_queries: int = 1000,
    seed: int = 0,
) -> tuple[int, float]:
    """Smallest power-of-two nprobe whose recall on sampled rows reaches ``target_recall``.

    The sampled rows' exact neighbours are found by brute force against all
    rows, then nprobe doubles from 1 until the target is met or nprobe reaches
    ``nlist``. Sets ``index.nprobe`` to the returned value.

    Returns
    -------
    nprobe : int
    estimated_recall : float
        Recall of the sampled rows at that nprobe.
    """
    _check_target_recall(target_recall)
    n = Xp.shape[0]
    _check_k(n, n_neighbors)
    rng = np.random.default_rng(seed)
    q = np.sort(rng.choice(n, size=min(n_queries, n), replace=False))
    true_I, _ = _graph(*_brute(Xp[q], Xp, n_neighbors, metric, _BATCH), metric, q)
    nprobe = 1
    while True:
        index.nprobe = nprobe
        I, _ = _graph(*_search(index, Xp[q], n_neighbors, _BATCH), metric, q)
        recall = knn_recall(I, true_I)
        if recall >= target_recall or nprobe >= index.nlist:
            return nprobe, recall
        nprobe = min(2 * nprobe, index.nlist)


class FaissKNNIndex:
    """Search index handed to UMAP as ``precomputed_knn[2]`` so ``transform`` works.

    Wraps any Faiss index over prepared data. ``query`` returns distances in
    the metric's true scale. Picklable through ``faiss.serialize_index``.

    Parameters
    ----------
    index : faiss.Index
        Index over the output of :func:`prepare`.
    metric : {"euclidean", "cosine"}
    neighbor_graph : tuple of ndarray
        The ``(indices, distances)`` graph of the indexed points.
    shift : ndarray of shape (d,), optional
        Column means that :func:`prepare` subtracted (Euclidean); new queries
        are shifted by the same vector. None means no shift.
    """

    def __init__(
        self,
        index: faiss.Index,
        metric: str,
        neighbor_graph: tuple[np.ndarray, np.ndarray],
        shift: np.ndarray | None = None,
    ):
        _check_metric(metric)
        self.index = index
        self.metric = metric
        self.neighbor_graph = neighbor_graph
        self.shift = None if shift is None or metric == "cosine" else np.asarray(shift, dtype=np.float32)
        self._angular_trees = metric == "cosine"

    @property
    def _raw_data(self) -> np.ndarray:
        """The indexed vectors, reconstructed from the index, with the shift added back."""
        if isinstance(self.index, faiss.IndexIVF):
            self.index.make_direct_map()
        data = self.index.reconstruct_n(0, self.index.ntotal)
        if self.shift is not None:
            data += self.shift
        return data

    def query(self, X: Any, k: int = 10, epsilon: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """k nearest indexed points of each row of ``X``; ``epsilon`` is ignored.

        Missing neighbours (possible with a small ``nprobe``) are ``-1`` / ``inf``,
        which UMAP skips.
        """
        Xq, _ = _prepare(X, self.metric, self.shift if self.shift is not None else np.zeros(self.index.d, np.float32))
        D, I = _search(self.index, Xq, k, _BATCH)
        return I.astype(_index_dtype(self.index.ntotal)), _true_distances(D, I, self.metric)

    def update(self, X: Any) -> None:
        raise NotImplementedError("eumap indexes cannot be updated; fit a new EUMAP on all the data.")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["index"] = faiss.serialize_index(self.index)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state["index"] = faiss.deserialize_index(state["index"])
        state.setdefault("shift", None)
        self.__dict__.update(state)


def ivf_knn(
    X: Any,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    *,
    nlist: int | None = None,
    nprobe: int | str = "auto",
    target_recall: float = 0.95,
    trainer: str = "emeans",
    trainer_kwds: dict[str, Any] | None = None,
    seed: int = 1234,
    n_threads: int | None = None,
) -> KNNGraph:
    """Approximate kNN graph of ``X`` with an e-means (or Lloyd) trained IVF-Flat index.

    Parameters
    ----------
    X : array-like of shape (n, d)
    n_neighbors : int
    metric : {"euclidean", "cosine"}
    nlist : int, optional
        Number of inverted lists; :func:`default_nlist` when None.
    nprobe : int or "auto"
        Lists probed per query; ``"auto"`` runs :func:`tune_nprobe`.
    target_recall : float
        Target of the nprobe tuning.
    trainer : {"emeans", "lloyd"}
    trainer_kwds : dict, optional
        Extra keywords for the trainer (see :func:`train_centroids`).
    seed : int
        Seeds the trainer and the tuning sample.
    n_threads : int, optional
        Faiss threads for this call (all CPUs in the affinity mask when None);
        the previous setting is restored afterwards.

    Returns
    -------
    KNNGraph
        ``indices``, ``distances`` (UMAP format, complete), ``index``
        (a :class:`FaissKNNIndex`) and ``info``.
    """
    t0 = time.perf_counter()
    # Validate everything before the (possibly long) training.
    _check_metric(metric)
    if trainer not in ("emeans", "lloyd"):
        raise ValueError(f"trainer must be 'emeans' or 'lloyd', got {trainer!r}.")
    auto = isinstance(nprobe, str) and nprobe == "auto"
    if not auto:
        _check_nprobe(nprobe, np.iinfo(np.int64).max)
    _check_target_recall(target_recall)
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}.")
    with faiss_threads(n_threads) as threads:
        Xp, shift = _prepare(X, metric)
        n = Xp.shape[0]
        _check_k(n, n_neighbors)
        nlist = default_nlist(n) if nlist is None else nlist
        _check_nlist(n, nlist)
        nlist = int(nlist)
        if not auto:
            _check_nprobe(nprobe, nlist)

        t = time.perf_counter()
        centroids = train_centroids(
            Xp, nlist, trainer, spherical=metric == "cosine", seed=seed, **(trainer_kwds or {})
        )
        t_train = time.perf_counter() - t

        t = time.perf_counter()
        index = build_ivf_index(Xp, centroids, metric)
        t_add = time.perf_counter() - t

        t = time.perf_counter()
        estimated_recall = None
        if auto:
            nprobe, estimated_recall = tune_nprobe(
                index, Xp, n_neighbors, metric, target_recall=target_recall, seed=seed
            )
        t_tune = time.perf_counter() - t

        t = time.perf_counter()
        I, D, n_repaired = _search_self(index, Xp, n_neighbors, nprobe, metric, _BATCH, True)
        t_search = time.perf_counter() - t

    info = {
        "method": "ivf",
        "trainer": trainer,
        "nlist": nlist,
        "nprobe": int(nprobe),
        "estimated_recall": estimated_recall,
        "n_repaired": n_repaired,
        "n_threads": threads,
        "t_train": t_train,
        "t_add": t_add,
        "t_tune": t_tune,
        "t_search": t_search,
        "t_total": time.perf_counter() - t0,
    }
    return KNNGraph(I, D, FaissKNNIndex(index, metric, (I, D), shift), info)


def exact_graph(X: Any, n_neighbors: int, metric: str = "euclidean", *, n_threads: int | None = None) -> KNNGraph:
    """Exact counterpart of :func:`ivf_knn`: brute-force graph plus a flat index for ``transform``."""
    t0 = time.perf_counter()
    with faiss_threads(n_threads) as threads:
        Xp, shift = _prepare(X, metric)
        _check_k(Xp.shape[0], n_neighbors)
        t = time.perf_counter()
        index = faiss.IndexFlatIP(Xp.shape[1]) if metric == "cosine" else faiss.IndexFlatL2(Xp.shape[1])
        index.add(Xp)
        t_add = time.perf_counter() - t
        t = time.perf_counter()
        I, D = _exact(Xp, n_neighbors, metric, _BATCH)
        t_search = time.perf_counter() - t
    info = {
        "method": "exact",
        "trainer": None,
        "nlist": None,
        "nprobe": None,
        "estimated_recall": 1.0,
        "n_repaired": 0,
        "n_threads": threads,
        "t_train": 0.0,
        "t_add": t_add,
        "t_tune": 0.0,
        "t_search": t_search,
        "t_total": time.perf_counter() - t0,
    }
    return KNNGraph(I, D, FaissKNNIndex(index, metric, (I, D), shift), info)
