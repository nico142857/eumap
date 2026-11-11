"""eumap: UMAP with a kNN graph from an e-means trained Faiss IVF index."""

from importlib.metadata import version

from ._umap import EUMAP
from .knn import (
    FaissKNNIndex,
    KNNGraph,
    build_ivf_index,
    default_nlist,
    exact_knn,
    ivf_knn,
    knn_recall,
    prepare,
    search_self,
    train_centroids,
    tune_nprobe,
)

__version__ = version("eumap")

__all__ = [
    "EUMAP",
    "FaissKNNIndex",
    "KNNGraph",
    "build_ivf_index",
    "default_nlist",
    "exact_knn",
    "ivf_knn",
    "knn_recall",
    "prepare",
    "search_self",
    "train_centroids",
    "tune_nprobe",
]
