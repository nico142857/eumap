import pickle
import warnings

import faiss
import numba
import numpy as np
import pytest
import scipy.sparse
from sklearn.exceptions import NotFittedError

import eumap

from conftest import blobs

# Small layouts keep the suite fast; compatibility_layout=False keeps seeded runs multi-threaded.
FAST = dict(n_epochs=50, compatibility_layout=False)


@pytest.fixture(scope="module")
def X_fit() -> np.ndarray:
    return blobs(6000, 16, seed=4)


@pytest.fixture(scope="module")
def ivf_model(X_fit) -> eumap.EUMAP:
    return eumap.EUMAP(exact_below=1000, nlist=64, random_state=0, **FAST).fit(X_fit)


def test_ivf_path(ivf_model, X_fit):
    emb = ivf_model.embedding_
    assert emb.shape == (len(X_fit), 2) and np.isfinite(emb).all()
    info = ivf_model.knn_info_
    assert info["method"] == "ivf" and info["nlist"] == 64 and info["estimated_recall"] >= 0.95
    assert ivf_model.umap_.n_neighbors == 15 and ivf_model.umap_.n_epochs == 50


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_exact_path(metric):
    X = blobs(2000, 16, seed=5)
    model = eumap.EUMAP(n_neighbors=10, metric=metric, random_state=0, **FAST)
    emb = model.fit_transform(X)
    assert emb.shape == (2000, 2) and np.isfinite(emb).all()
    assert model.knn_info_["method"] == "exact"
    I, D = eumap.exact_knn(X, 10, metric)
    np.testing.assert_array_equal(model.umap_.knn_indices, I)


def test_transform_and_pickle(ivf_model):
    X_new = blobs(300, 16, seed=4)[:300] + 0.01
    emb = ivf_model.transform(X_new)
    assert emb.shape == (300, 2) and np.isfinite(emb).all()

    clone = pickle.loads(pickle.dumps(ivf_model))
    np.testing.assert_array_equal(clone.embedding_, ivf_model.embedding_)
    emb2 = clone.transform(X_new)
    assert emb2.shape == (300, 2) and np.isfinite(emb2).all()
    np.testing.assert_allclose(emb2, emb, atol=1e-3)


def test_seeded_fit_is_deterministic(X_fit):
    a = eumap.EUMAP(exact_below=1000, nlist=64, random_state=3, **FAST).fit_transform(X_fit)
    b = eumap.EUMAP(exact_below=1000, nlist=64, random_state=3, **FAST).fit_transform(X_fit)
    np.testing.assert_array_equal(a, b)


def test_fit_contract(X_fit):
    """No misleading warning, UMAP gets copies, Faiss and numba thread counts are restored."""
    faiss.omp_set_num_threads(3)
    numba.set_num_threads(3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = eumap.EUMAP(exact_below=1000, nlist=64, n_jobs=2, **FAST).fit(X_fit)
    assert not [w for w in caught if "precomputed_knn" in str(w.message)], [str(w.message) for w in caught]
    assert faiss.omp_get_max_threads() == 3 and numba.get_num_threads() == 3
    assert model.knn_info_["n_threads"] == 2
    index = model.umap_._knn_search_index
    assert isinstance(index, eumap.FaissKNNIndex)
    assert not np.shares_memory(model.umap_._knn_indices, index.neighbor_graph[0])
    assert not np.shares_memory(model.umap_._knn_dists, index.neighbor_graph[1])

    # UMAP fails on a wrong-length y; the thread counts must still be restored.
    with pytest.raises(ValueError):
        eumap.EUMAP(n_jobs=2, **FAST).fit(X_fit[:2000], y=np.zeros(5))
    assert faiss.omp_get_max_threads() == 3 and numba.get_num_threads() == 3


def test_unseeded_graph_is_deterministic(X_fit):
    """random_state=None leaves UMAP unseeded but the trainer uses a fixed seed."""
    a = eumap.EUMAP(exact_below=1000, nlist=64, n_epochs=2, compatibility_layout=False).fit(X_fit)
    b = eumap.EUMAP(exact_below=1000, nlist=64, n_epochs=2, compatibility_layout=False).fit(X_fit)
    np.testing.assert_array_equal(a.umap_.knn_indices, b.umap_.knn_indices)
    assert a.knn_info_["nprobe"] == b.knn_info_["nprobe"]


def test_exact_below_boundary():
    X = blobs(1200, 8, seed=6)
    kw = dict(nlist=16, n_epochs=2, compatibility_layout=False)
    assert eumap.EUMAP(exact_below=1200, **kw).fit(X).knn_info_["method"] == "ivf"
    assert eumap.EUMAP(exact_below=1201, **kw).fit(X).knn_info_["method"] == "exact"


def test_offset_data_exact_path():
    """The exact path is translation invariant (Faiss' BLAS form is not, without centring)."""
    rng = np.random.default_rng(0)
    X = (rng.normal(scale=3.0, size=(40, 128))[rng.integers(40, size=3000)] + rng.normal(size=(3000, 128)))
    X = X.astype(np.float32)
    a = eumap.EUMAP(n_epochs=2, compatibility_layout=False).fit(X)
    b = eumap.EUMAP(n_epochs=2, compatibility_layout=False).fit(X + np.float32(3000))
    assert eumap.knn_recall(b.umap_.knn_indices, a.umap_.knn_indices) > 0.999
    assert np.all(b.umap_.knn_dists[:, 1] > 0)
    np.testing.assert_allclose(b.umap_.knn_dists, a.umap_.knn_dists, rtol=1e-3, atol=1e-3)


def test_estimator_errors():
    X = blobs(500, 8)
    with pytest.raises(ValueError, match="umap.UMAP"):
        eumap.EUMAP(metric="manhattan")
    with pytest.raises(TypeError, match="precomputed_knn"):
        eumap.EUMAP(precomputed_knn=(None, None))
    with pytest.raises(TypeError, match="random_state"):
        eumap.EUMAP(random_state=np.random.RandomState(0))
    with pytest.raises(ValueError, match="n_jobs"):
        eumap.EUMAP(n_jobs=0)
    with pytest.raises(TypeError, match="n_jobs"):
        eumap.EUMAP(n_jobs=None)
    with pytest.raises(ValueError, match="random_state"):
        eumap.EUMAP(random_state=-1)
    with pytest.raises(ValueError, match="random_state"):
        eumap.EUMAP(random_state=2**32)
    with pytest.raises(TypeError, match="n_epoch"):
        eumap.EUMAP(n_epoch=10)  # misspelt UMAP keyword: fails at construction
    with pytest.raises(ValueError, match="unique"):
        eumap.EUMAP(unique=True)
    with pytest.raises(ValueError, match="NUMBA_NUM_THREADS"):
        eumap.EUMAP(n_jobs=numba.config.NUMBA_NUM_THREADS + 1).fit(X)
    with pytest.raises(ValueError, match="at least 39 samples"):
        eumap.EUMAP(exact_below=0).fit(X[:30])
    with pytest.raises(NotFittedError):
        eumap.EUMAP().transform(X)
    with pytest.raises(TypeError, match="dense"):
        eumap.EUMAP().fit(scipy.sparse.csr_matrix(X))
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        eumap.EUMAP().fit(X_nan)
    with pytest.raises(ValueError, match="points per list"):
        eumap.EUMAP(exact_below=0, nlist=100).fit(X)
