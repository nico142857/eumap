"""The EUMAP estimator: a Faiss kNN graph handed to an unmodified ``umap.UMAP``."""

from __future__ import annotations

import numbers
import warnings
from typing import Any

import numba
import numpy as np
import umap
from sklearn.exceptions import NotFittedError

from .knn import _check_metric, available_cpus, exact_graph, ivf_knn

DEFAULT_SEED = 1234


class EUMAP:
    """UMAP whose kNN graph comes from an e-means trained Faiss IVF-Flat index.

    Parameters
    ----------
    n_neighbors : int
        Passed to UMAP and used for the kNN graph.
    metric : {"euclidean", "cosine"}
        Other metrics raise ``ValueError``; use ``umap.UMAP`` for them.
    nlist, nprobe, target_recall, trainer
        See :func:`eumap.ivf_knn`.
    exact_below : int
        Below this many samples the graph is computed exactly by brute force
        instead of with an IVF index.
    random_state : int or None
        An int in ``[0, 2**32)`` seeds the trainer, the nprobe tuning sample
        and UMAP. ``None`` leaves UMAP unseeded and uses a fixed seed for the
        trainer.
    n_jobs : int
        Threads for Faiss and UMAP; -1 means all CPUs in the affinity mask.
        At most numba's ``NUMBA_NUM_THREADS``.
    **umap_kwds
        Passed unchanged to ``umap.UMAP``. Unknown keywords raise ``TypeError``
        here; ``precomputed_knn`` and ``unique=True`` are not allowed.

    Attributes
    ----------
    umap_ : umap.UMAP
        The fitted UMAP model.
    embedding_ : ndarray of shape (n_samples, n_components)
    knn_info_ : dict
        How the graph was built (see :func:`eumap.ivf_knn`); ``method`` is
        ``"exact"`` or ``"ivf"``.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        *,
        nlist: int | None = None,
        nprobe: int | str = "auto",
        target_recall: float = 0.95,
        trainer: str = "emeans",
        exact_below: int = 100_000,
        random_state: int | None = None,
        n_jobs: int = -1,
        **umap_kwds: Any,
    ):
        _check_metric(metric)
        if "precomputed_knn" in umap_kwds:
            raise TypeError("EUMAP computes precomputed_knn itself; do not pass it.")
        if umap_kwds.get("unique", False):
            raise ValueError(
                "EUMAP does not support unique=True (UMAP cannot combine it with a precomputed kNN "
                "graph); remove duplicate rows from X yourself, or use umap.UMAP."
            )
        if random_state is not None:
            if isinstance(random_state, bool) or not isinstance(random_state, numbers.Integral):
                raise TypeError(f"random_state must be an int or None, got {type(random_state).__name__}.")
            if not 0 <= random_state < 2**32:
                raise ValueError(f"random_state must be in [0, 2**32), got {random_state}.")
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, numbers.Integral):
            raise TypeError(f"n_jobs must be an int (-1 for all CPUs), got {n_jobs!r}.")
        if n_jobs != -1 and n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or >= 1, got {n_jobs}.")
        # Fail now on misspelt UMAP keywords rather than after the kNN stage.
        umap.UMAP(n_neighbors=n_neighbors, metric=metric, random_state=random_state, n_jobs=n_jobs, **umap_kwds)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.target_recall = target_recall
        self.trainer = trainer
        self.exact_below = exact_below
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.umap_kwds = umap_kwds

    def fit(self, X: Any, y: Any = None) -> "EUMAP":
        """Build the kNN graph of ``X`` and fit UMAP on it. Returns ``self``."""
        if self.n_jobs > numba.config.NUMBA_NUM_THREADS:
            raise ValueError(
                f"n_jobs={self.n_jobs} exceeds numba's thread limit NUMBA_NUM_THREADS="
                f"{numba.config.NUMBA_NUM_THREADS}; use n_jobs=-1 or a smaller value."
            )
        n_threads = available_cpus() if self.n_jobs == -1 else self.n_jobs
        seed = DEFAULT_SEED if self.random_state is None else int(self.random_state)
        if np.shape(X)[0] < self.exact_below:
            graph = exact_graph(X, self.n_neighbors, self.metric, n_threads=n_threads)
        else:
            graph = ivf_knn(
                X,
                self.n_neighbors,
                self.metric,
                nlist=self.nlist,
                nprobe=self.nprobe,
                target_recall=self.target_recall,
                trainer=self.trainer,
                seed=seed,
                n_threads=n_threads,
            )
        model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            # UMAP writes -1/inf into these arrays in place.
            precomputed_knn=(graph.indices.copy(), graph.distances.copy(), graph.index),
            **self.umap_kwds,
        )
        numba_threads = numba.get_num_threads()
        try:
            with warnings.catch_warnings():
                # Wrong for our index: transform works through FaissKNNIndex.query.
                warnings.filterwarnings("ignore", message=r"precomputed_knn\[2\]")
                model.fit(X, y)
        finally:
            # UMAP restores numba's thread count only when fit succeeds.
            numba.set_num_threads(numba_threads)
        self.umap_ = model
        self.embedding_ = model.embedding_
        self.knn_info_ = graph.info
        return self

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit and return the embedding of ``X``."""
        return self.fit(X, y).embedding_

    def transform(self, X: Any) -> np.ndarray:
        """Embed new points into the fitted embedding (``umap.UMAP.transform``)."""
        if not hasattr(self, "umap_"):
            raise NotFittedError("This EUMAP instance is not fitted yet; call fit first.")
        return self.umap_.transform(X)
