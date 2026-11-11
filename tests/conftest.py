import numpy as np
import pytest


def blobs(n: int, d: int, n_centers: int = 20, seed: int = 0) -> np.ndarray:
    """Gaussian clusters with distinct, untied distances."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(n_centers, d))
    labels = rng.integers(n_centers, size=n)
    return (centers[labels] + rng.normal(size=(n, d))).astype(np.float32)


@pytest.fixture(scope="session")
def X_small() -> np.ndarray:
    return blobs(3000, 16)


@pytest.fixture(scope="session")
def X_medium() -> np.ndarray:
    return blobs(20000, 16, n_centers=50, seed=1)
