import pickle
import tracemalloc

import faiss
import numpy as np
import pytest
import scipy.sparse
from sklearn.neighbors import NearestNeighbors

import eumap
from eumap.knn import exact_graph

from conftest import blobs

K = 15


def assert_umap_format(I: np.ndarray, D: np.ndarray, n: int, k: int = K) -> None:
    assert I.shape == D.shape == (n, k)
    assert I.dtype == np.int32 and D.dtype == np.float32
    assert I.flags.c_contiguous and D.flags.c_contiguous
    np.testing.assert_array_equal(I[:, 0], np.arange(n))
    assert np.all(D[:, 0] == 0)
    assert np.all(D[:, 1:] >= D[:, :-1])  # sorted


def sklearn_knn(X: np.ndarray, k: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric).fit(X.astype(np.float64))
    D, I = nn.kneighbors(X.astype(np.float64))
    return I, D


# --- building blocks -------------------------------------------------------


def test_prepare_formats_and_never_modifies_input(X_small):
    Xf = np.asfortranarray(X_small.astype(np.float64))
    before = Xf.copy()
    Xp = eumap.prepare(Xf, "euclidean")
    assert Xp.dtype == np.float32 and Xp.flags.c_contiguous
    np.testing.assert_array_equal(Xf, before)
    np.testing.assert_allclose(Xp, before - before.mean(axis=0), atol=1e-5)  # centred copy
    np.testing.assert_allclose(Xp.mean(axis=0), 0.0, atol=1e-5)
    X32 = X_small.copy()
    assert not np.shares_memory(eumap.prepare(X32, "euclidean"), X32)
    np.testing.assert_array_equal(X32, X_small)
    X = X_small.copy()
    Xc = eumap.prepare(X, "cosine")
    np.testing.assert_array_equal(X, X_small)
    np.testing.assert_allclose(np.linalg.norm(Xc, axis=1), 1.0, rtol=1e-5)


def test_default_nlist():
    assert eumap.default_nlist(100_000) == 1024
    assert eumap.default_nlist(1_000_000) == 4096
    assert eumap.default_nlist(1000) == 1000 // 39
    assert eumap.default_nlist(10) == 1


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_format_against_sklearn(X_small, metric):
    n = len(X_small)
    I_ref, D_ref = sklearn_knn(X_small, K, metric)

    I, D = eumap.exact_knn(X_small, K, metric)
    assert_umap_format(I, D, n)
    assert eumap.knn_recall(I, I_ref) > 0.999
    np.testing.assert_allclose(D, D_ref, atol=2e-3)

    g = eumap.ivf_knn(X_small, K, metric, nlist=32, nprobe=32, seed=0)  # nprobe = nlist: exact
    assert_umap_format(g.indices, g.distances, n)
    assert eumap.knn_recall(g.indices, I_ref) > 0.999
    np.testing.assert_allclose(g.distances, D_ref, atol=2e-3)
    assert set(g.info) == {
        "method", "trainer", "nlist", "nprobe", "estimated_recall", "n_repaired",
        "n_threads", "t_train", "t_add", "t_tune", "t_search", "t_total",
    }  # fmt: skip
    assert g.info["method"] == "ivf" and g.info["nlist"] == 32 and g.info["nprobe"] == 32


def offset_clusters(n: int = 4000, d: int = 128, offset: float = 1000.0, seed: int = 0) -> np.ndarray:
    """Clustered data far from the origin: Faiss' BLAS distance form loses precision here."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(40, d))
    X = centers[rng.integers(40, size=n)] + rng.normal(size=(n, d))
    return (X + offset).astype(np.float32)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_duplicates_keep_self_first(metric):
    base = blobs(500, 32, seed=3) * 30  # SIFT-like magnitudes
    X = np.vstack([base, base, base, np.repeat(base[:1], 30, axis=0)])  # 30 copies > K
    n = len(X)
    I, D = eumap.exact_knn(X, K, metric)
    assert_umap_format(I, D, n)
    # Every row of the first 1500 has two exact copies; the 30 copies of row 0 fill whole rows.
    assert np.all(D[:1500, 1:3] == 0) and np.all(D[1500:, 1:] == 0)
    for i in range(1, 500):
        assert set(I[i, :3]) == {i, i + 500, i + 1000}
    g = eumap.ivf_knn(X, K, metric, nlist=16, nprobe=4, seed=0)
    assert_umap_format(g.indices, g.distances, n)
    # The 30 copies of row 0 are each other's neighbours. Faiss' IVF scan computes L2 directly (exact 0);
    # for cosine it returns 1 - <x, x>, which is only zero up to float32 resolution.
    tol = 0.0 if metric == "euclidean" else 1e-6
    assert np.all(g.distances[1500:, 1:] <= tol)


def test_offset_data_is_exact():
    """A large common offset must not change neighbours or distances (Euclidean is translation invariant)."""
    X = offset_clusters()
    I_ref, _ = sklearn_knn(X, K, "euclidean")
    I, D = eumap.exact_knn(X, K)
    assert_umap_format(I, D, len(X))
    assert eumap.knn_recall(I, I_ref) > 0.999
    X64 = X.astype(np.float64)  # direct differences: sklearn's own distances carry the same BLAS-form error
    D_ref = np.linalg.norm(X64[I] - X64[:, None, :], axis=2)
    np.testing.assert_allclose(D, D_ref, rtol=1e-4, atol=1e-4)
    assert np.all(D[:, 1] > 0)

    # The tuning ground truth is exact too, so the estimated recall matches the true one.
    g = eumap.ivf_knn(X, K, nlist=64, target_recall=0.9, seed=0)
    true_recall = eumap.knn_recall(g.indices, I_ref)
    assert abs(g.info["estimated_recall"] - true_recall) < 0.03, (g.info, true_recall)
    assert g.info["nprobe"] < 64

    # New queries are moved by the training mean, not by their own.
    Xq = offset_clusters(n=200, seed=1)
    _, I_q = NearestNeighbors(n_neighbors=K).fit(X64).kneighbors(Xq.astype(np.float64))
    for index in (exact_graph(X, K).index, g.index):
        if isinstance(index.index, faiss.IndexIVF):
            index.index.nprobe = 64
        I2, D2 = index.query(Xq, K)
        assert eumap.knn_recall(I2, I_q) > 0.999
        D_q = np.linalg.norm(X64[I2] - Xq.astype(np.float64)[:, None, :], axis=2)
        np.testing.assert_allclose(D2, D_q, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(g.index._raw_data, X, atol=1e-3)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_missing_neighbours_are_repaired(X_small, metric):
    k = 60  # more than the average list size (about 3000 / 76 points): nprobe=1 must miss
    Xp = eumap.prepare(X_small, metric)
    nlist = len(Xp) // 39
    index = eumap.build_ivf_index(Xp, eumap.train_centroids(Xp, nlist, spherical=metric == "cosine"), metric)
    I, D = eumap.search_self(index, Xp, k, 1, metric, repair=False)
    missing = I < 0
    assert missing.any()
    assert np.all(np.isinf(D[missing])) and np.all(np.isfinite(D[~missing]))
    assert np.all(D[:, 1:] >= D[:, :-1])  # sorted, missing slots last

    I, D = eumap.search_self(index, Xp, k, 1, metric, repair=True)
    assert_umap_format(I, D, len(Xp), k)
    assert np.all(I >= 0) and np.all(np.isfinite(D))
    g = eumap.ivf_knn(X_small, k, metric, nlist=nlist, nprobe=1)
    assert g.info["n_repaired"] == int(missing.any(axis=1).sum())
    rows = np.flatnonzero(missing.any(axis=1))
    I_exact, _ = eumap.exact_knn(X_small, k, metric)
    assert eumap.knn_recall(g.indices[rows], I_exact[rows]) > 0.999


def test_tune_nprobe_reaches_target(X_medium):
    Xp = eumap.prepare(X_medium, "euclidean")
    index = eumap.build_ivf_index(Xp, eumap.train_centroids(Xp, 256), "euclidean")
    nprobe, est = eumap.tune_nprobe(index, Xp, K, "euclidean", target_recall=0.95, seed=1234)
    assert est >= 0.95 and 1 < nprobe <= 256
    lower, _ = eumap.tune_nprobe(index, Xp, K, "euclidean", target_recall=0.5, seed=1234)
    assert lower < nprobe
    I, _ = eumap.search_self(index, Xp, K, nprobe, "euclidean")
    I_exact, _ = eumap.exact_knn(Xp, K)
    assert eumap.knn_recall(I, I_exact) >= 0.93

    g = eumap.ivf_knn(X_medium, K, nlist=256, target_recall=0.95)
    assert g.info["nprobe"] == nprobe and g.info["estimated_recall"] == est


def test_trainers_give_similar_recall(X_medium):
    I_exact, _ = eumap.exact_knn(X_medium, K)
    recall = {
        trainer: eumap.knn_recall(eumap.ivf_knn(X_medium, K, nlist=256, nprobe=4, trainer=trainer).indices, I_exact)
        for trainer in ("emeans", "lloyd")
    }
    assert abs(recall["emeans"] - recall["lloyd"]) < 0.03, recall


def test_trainer_kwds_are_forwarded(X_small):
    Xp = eumap.prepare(X_small, "euclidean")
    a = eumap.train_centroids(Xp, 32, "emeans", nepochs=2)
    b = eumap.train_centroids(Xp, 32, "emeans")
    assert a.shape == b.shape == (32, 16) and not np.array_equal(a, b)
    c = eumap.train_centroids(Xp, 32, "lloyd", niter=2)
    d = eumap.train_centroids(Xp, 32, "lloyd")
    assert c.shape == (32, 16) and not np.array_equal(c, d)


@pytest.mark.parametrize("trainer", ["emeans", "lloyd"])
def test_deterministic_for_fixed_seed(X_medium, trainer):
    a = eumap.ivf_knn(X_medium, K, nlist=128, trainer=trainer, seed=7)
    b = eumap.ivf_knn(X_medium, K, nlist=128, trainer=trainer, seed=7)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.distances, b.distances)
    assert a.info["nprobe"] == b.info["nprobe"]


def test_threads_are_restored_even_on_error(X_small):
    faiss.omp_set_num_threads(3)
    g = eumap.ivf_knn(X_small, K, nlist=32, nprobe=2, n_threads=2)
    assert g.info["n_threads"] == 2 and faiss.omp_get_max_threads() == 3
    with pytest.raises(ValueError):
        eumap.ivf_knn(X_small, K, nlist=len(X_small), n_threads=1)  # fails inside the thread block
    assert faiss.omp_get_max_threads() == 3


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_index_query_and_pickle(X_small, metric):
    g = eumap.ivf_knn(X_small, K, metric, nlist=32, nprobe=32)
    Xq = blobs(50, 16, seed=9)
    I, D = g.index.query(Xq, K, epsilon=0.12)
    D_ref, I_ref = NearestNeighbors(n_neighbors=K, metric=metric).fit(X_small).kneighbors(Xq)
    assert eumap.knn_recall(I, I_ref) > 0.99
    np.testing.assert_allclose(D, D_ref, atol=2e-3)
    assert I.dtype == np.int32 and D.dtype == np.float32
    assert g.index._angular_trees == (metric == "cosine")
    assert g.index._raw_data.shape == X_small.shape
    assert g.index.neighbor_graph[0] is g.indices

    clone = pickle.loads(pickle.dumps(g.index))
    I2, D2 = clone.query(Xq, K)
    np.testing.assert_array_equal(I, I2)
    np.testing.assert_array_equal(D, D2)
    with pytest.raises(NotImplementedError):
        g.index.update(Xq)

    e = exact_graph(X_small, K, metric)
    I3, D3 = e.index.query(Xq, K)
    np.testing.assert_allclose(D3, D_ref, atol=2e-3)


def test_knn_recall_large_k_matches_sets_and_stays_small():
    rng = np.random.default_rng(0)
    n, k = 20000, 200
    true = np.argsort(rng.random((n, 400)), axis=1)[:, :k].astype(np.int32)
    approx = true.copy()
    approx[:, 1::3] = rng.integers(400, 800, size=approx[:, 1::3].shape)  # misses
    approx[::7, -1] = -1
    expected = np.mean([len(set(a[1:]) & set(b[1:]) - {-1}) for a, b in zip(approx[:500], true[:500])]) / (k - 1)
    assert eumap.knn_recall(approx[:500], true[:500]) == pytest.approx(expected)
    tracemalloc.start()
    eumap.knn_recall(approx, true)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 100 * 2**20, f"knn_recall peak {peak / 2**20:.0f} MB"


def test_lloyd_accepts_wide_seeds(X_small):
    Xp = eumap.prepare(X_small, "euclidean")
    a = eumap.train_centroids(Xp, 16, "lloyd", seed=2**31 + 5)
    np.testing.assert_array_equal(a, eumap.train_centroids(Xp, 16, "lloyd", seed=5))
    g = eumap.ivf_knn(X_small, K, nlist=16, nprobe=2, trainer="lloyd", seed=2**32 - 1)
    assert g.indices.shape == (len(X_small), K)


@pytest.mark.parametrize(
    "kwds, match",
    [
        (dict(nprobe=0), "nprobe"),
        (dict(nprobe=10**6), "nprobe"),
        (dict(nprobe=2.5), "nprobe"),
        (dict(nprobe="fast"), "nprobe"),
        (dict(target_recall=95), "target_recall"),
        (dict(target_recall=0.0), "target_recall"),
        (dict(trainer="kmeans++"), "trainer"),
        (dict(nlist=10**6), "points per list"),
        (dict(seed=-1), "seed"),
    ],
)
def test_ivf_knn_validates_before_training(X_small, monkeypatch, kwds, match):
    def no_training(*args, **kw):
        raise AssertionError("training started before the parameters were validated")

    monkeypatch.setattr("eumap.knn.train_centroids", no_training)
    with pytest.raises(ValueError, match=match):
        eumap.ivf_knn(X_small, K, **{"nlist": 32, **kwds})


# --- errors ------------------------------------------------------------------


def test_errors(X_small):
    with pytest.raises(ValueError, match="umap.UMAP"):
        eumap.prepare(X_small, "manhattan")
    with pytest.raises(ValueError, match="umap.UMAP"):
        eumap.ivf_knn(X_small, metric="correlation")
    with pytest.raises(ValueError, match="umap.UMAP"):
        eumap.exact_knn(X_small, K, metric="l1")
    with pytest.raises(TypeError, match="dense"):
        eumap.prepare(scipy.sparse.csr_matrix(X_small), "euclidean")
    bad = X_small.copy()
    bad[5, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        eumap.prepare(bad, "euclidean")
    bad[5, 3] = np.inf
    with pytest.raises(ValueError, match="NaN"):
        eumap.ivf_knn(bad)
    with pytest.raises(ValueError, match="2-D"):
        eumap.prepare(X_small[0], "euclidean")

    Xp = eumap.prepare(X_small, "euclidean")
    for nlist in (0, len(Xp) // 39 + 1):
        with pytest.raises(ValueError, match="points per list"):
            eumap.train_centroids(Xp, nlist)
    with pytest.raises(ValueError, match="trainer"):
        eumap.train_centroids(Xp, 8, "kmeans++")
    with pytest.raises(ValueError, match="dimension"):
        eumap.build_ivf_index(Xp, np.zeros((8, 3), np.float32), "euclidean")
    index = eumap.build_ivf_index(Xp, eumap.train_centroids(Xp, 8), "euclidean")
    with pytest.raises(ValueError, match="nprobe"):
        eumap.search_self(index, Xp, K, 9, "euclidean")
    with pytest.raises(ValueError, match="nprobe"):
        eumap.ivf_knn(X_small, nlist=8, nprobe="fast")
    for k in (1, len(Xp) + 1):
        with pytest.raises(ValueError, match="n_neighbors"):
            eumap.exact_knn(X_small, k)
    with pytest.raises(ValueError, match="target_recall"):
        eumap.tune_nprobe(index, Xp, K, "euclidean", target_recall=1.5)
    with pytest.raises(ValueError, match="shape"):
        eumap.knn_recall(np.zeros((5, 3)), np.zeros((5, 4)))
    with pytest.raises(ValueError, match="n_threads"):
        eumap.ivf_knn(X_small, n_threads=0)
    with pytest.raises(ValueError, match="at least 39 samples"):
        eumap.ivf_knn(X_small[:30], 5)
    with pytest.raises(ValueError, match="batch_size"):
        eumap.search_self(index, Xp, K, 2, "euclidean", batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        eumap.exact_knn(X_small, K, batch_size=0)
