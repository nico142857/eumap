"""Benchmark datasets: download, cache under ``data/``, verify and load.

Every loader returns a :class:`Dataset` whose ``X`` is float32 and C-contiguous.
Downloads only ever go to ``data/<name>/raw/`` and are parsed as data (IDX
headers with ``struct``, CSV with ``numpy.loadtxt``, HDF5 with ``h5py``); nothing
downloaded is executed. The parsed array is cached as ``data/<name>/X.npy``
(and ``y.npy`` for labels) together with ``meta.json`` (source URLs, checksums,
shape), and checked on every load: expected shape, float32, all finite.

Usage as a script (inside a Slurm job)::

    python -I -u benchmarks/datasets.py mnist fashion_mnist covtype ...
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path("data")

CORE = ["synthetic", "mnist", "fashion_mnist", "covtype", "sift1m", "glove100"]
EXTENDED = ["gist1m", "deep10m"]

# Synthetic sweep: mixture of C low-dimensional Gaussian subspaces in d dimensions,
# x = mu_c + z @ B_c + sigma * e,  mu_c ~ N(0, mean_scale^2 I_d), B_c ~ N(0, 1)^(sub_dim x d),
# z ~ N(0, I_sub_dim), e ~ N(0, I_d). Rows are generated in fixed chunks with their own
# seed so every prefix of length n is identical whatever the total size (nested prefixes).
SYNTH = dict(n_max=3_000_000, d=128, n_components=100, sub_dim=10, noise=1.0, mean_scale=1.0, seed=20241111)
SYNTH_SIZES = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
_SYNTH_CHUNK = 65_536

_MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
_FASHION_MIRRORS = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]
# Published MD5 sums of the IDX files (torchvision's resource lists).
_MNIST_MD5 = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
_FASHION_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
# scikit-learn's covtype fetcher uses this file and SHA256.
_COVTYPE = ("https://ndownloader.figshare.com/files/5976039", "covtype.data.gz",
            "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771")
_ANN = "http://ann-benchmarks.com/"

# name -> (expected shape or None if only the column count is known, metric, has labels)
SPECS: dict[str, dict[str, Any]] = {
    "synthetic": dict(shape=(SYNTH["n_max"], SYNTH["d"]), metric="euclidean", labels=True),
    "mnist": dict(shape=(70_000, 784), metric="euclidean", labels=True),
    "fashion_mnist": dict(shape=(70_000, 784), metric="euclidean", labels=True),
    "covtype": dict(shape=(581_012, 54), metric="euclidean", labels=True),
    "sift1m": dict(shape=(1_000_000, 128), metric="euclidean", labels=False, hdf5="sift-128-euclidean.hdf5"),
    "glove100": dict(shape=(None, 100), metric="cosine", labels=False, hdf5="glove-100-angular.hdf5"),
    "gist1m": dict(shape=(1_000_000, 960), metric="euclidean", labels=False, hdf5="gist-960-euclidean.hdf5"),
    "deep10m": dict(shape=(None, 96), metric="cosine", labels=False, hdf5="deep-image-96-angular.hdf5",
                    full_gt=False),
}


@dataclass
class Dataset:
    name: str
    X: np.ndarray  # float32, C-contiguous
    y: np.ndarray | None
    metric: str
    n_full: int
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def full_gt(self) -> bool:
        return SPECS[self.name].get("full_gt", True)


# ----------------------------------------------------------------------------- helpers

def _log(msg: str) -> None:
    print(f"[datasets {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _download(urls: list[str], dest: Path, *, retries: int = 5) -> str:
    """Stream the first URL that works to ``dest`` (resuming a partial ``.part``); return the URL used."""
    if dest.exists():
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f"{dest.name}.part{os.getpid()}")
    last_err: Exception | None = None
    for url in urls:
        for attempt in range(retries):
            try:
                have = part.stat().st_size if part.exists() else 0
                req = urllib.request.Request(url, headers={"User-Agent": "eumap-benchmark/0.1"})
                if have:
                    req.add_header("Range", f"bytes={have}-")
                with urllib.request.urlopen(req, timeout=120) as r:
                    if have and r.status != 206:  # server ignored the range: start over
                        have = 0
                    total = r.headers.get("Content-Length")
                    _log(f"GET {url} -> {dest} (status {r.status}, {total} bytes, resume at {have})")
                    with open(part, "ab" if have else "wb") as f:
                        while block := r.read(1 << 23):
                            f.write(block)
                os.replace(part, dest)
                return url
            except Exception as err:  # noqa: BLE001 - network errors of all kinds: retry
                last_err = err
                _log(f"download error ({url}, attempt {attempt + 1}): {err!r}")
                time.sleep(min(60, 5 * 2**attempt))
    raise RuntimeError(f"could not download {dest.name}: {last_err!r}")


def _verify(name: str, X: np.ndarray, y: np.ndarray | None) -> None:
    exp_n, exp_d = SPECS[name]["shape"]
    if X.ndim != 2 or X.shape[1] != exp_d or (exp_n is not None and X.shape[0] != exp_n):
        raise ValueError(f"{name}: unexpected shape {X.shape}, expected ({exp_n}, {exp_d}).")
    if X.dtype != np.float32 or not X.flags.c_contiguous:
        raise ValueError(f"{name}: X must be float32 C-contiguous, got {X.dtype}.")
    for s in range(0, X.shape[0], 1 << 20):
        if not np.isfinite(X[s : s + (1 << 20)]).all():
            raise ValueError(f"{name}: X contains non-finite values.")
    if y is not None and y.shape != (X.shape[0],):
        raise ValueError(f"{name}: labels have shape {y.shape}, expected ({X.shape[0]},).")


def _save(name: str, X: np.ndarray, y: np.ndarray | None, meta: dict[str, Any], root: Path) -> None:
    _verify(name, X, y)
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"X.npy.tmp{os.getpid()}.npy"
    np.save(tmp, X)
    os.replace(tmp, d / "X.npy")
    if y is not None:
        tmp = d / f"y.npy.tmp{os.getpid()}.npy"
        np.save(tmp, y)
        os.replace(tmp, d / "y.npy")
    meta = dict(meta, shape=list(X.shape), dtype=str(X.dtype), labels=y is not None,
                n_classes=int(np.unique(y).size) if y is not None else None,
                created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


# ----------------------------------------------------------------------------- builders

def _read_idx(path: Path, md5: str) -> np.ndarray:
    got = _hash(path, "md5")
    if got != md5:
        raise ValueError(f"{path}: md5 {got} != published {md5}; delete it and retry.")
    with gzip.open(path, "rb") as f:
        raw = f.read()
    zero, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    if zero != 0 or dtype_code != 0x08:
        raise ValueError(f"{path}: not an unsigned-byte IDX file.")
    shape = struct.unpack(">" + "I" * ndim, raw[4 : 4 + 4 * ndim])
    arr = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"{path}: size {arr.size} does not match header {shape}.")
    return arr.reshape(shape)


def _build_idx(name: str, root: Path) -> None:
    mirrors, md5s = (_MNIST_MIRRORS, _MNIST_MD5) if name == "mnist" else (_FASHION_MIRRORS, _FASHION_MD5)
    raw = root / name / "raw"
    used = {f: _download([m + f for m in mirrors], raw / f) for f in md5s}
    imgs = [_read_idx(raw / f, md5s[f]) for f in ("train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz")]
    labs = [_read_idx(raw / f, md5s[f]) for f in ("train-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz")]
    X = np.ascontiguousarray(np.concatenate(imgs).reshape(-1, 784), dtype=np.float32)
    y = np.concatenate(labs).astype(np.int64)
    _save(name, X, y, {"source": used, "md5": md5s, "preprocessing": "raw pixels 0..255 as float32, train then test"}, root)


def _build_covtype(root: Path) -> None:
    url, fname, sha = _COVTYPE
    path = root / "covtype" / "raw" / fname
    _download([url], path)
    got = _hash(path, "sha256")
    if got != sha:
        raise ValueError(f"{path}: sha256 {got} != {sha}; delete it and retry.")
    with gzip.open(path, "rt") as f:
        data = np.loadtxt(f, delimiter=",", dtype=np.float64)
    X, y = data[:, :-1], data[:, -1].astype(np.int64)
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    X = np.ascontiguousarray(X, dtype=np.float32)
    _save("covtype", X, y, {"source": url, "sha256": sha,
                            "preprocessing": "each of the 54 columns standardised to mean 0, sd 1 (all rows)"}, root)


def _build_hdf5(name: str, root: Path) -> None:
    import h5py

    fname = SPECS[name]["hdf5"]
    path = root / name / "raw" / fname
    used = _download([_ANN + fname], path)
    with h5py.File(path, "r") as f:
        ds = f["train"]
        X = np.empty(ds.shape, dtype=np.float32)
        for s in range(0, ds.shape[0], 1 << 20):
            X[s : s + (1 << 20)] = ds[s : s + (1 << 20)]
    meta: dict[str, Any] = {"source": used if used != "cached" else _ANN + fname,
                            "sha256": _hash(path, "sha256"), "hdf5_field": "train"}
    if SPECS[name]["metric"] == "cosine":
        norms = np.linalg.norm(X, axis=1)
        zero = norms == 0
        meta["zero_norm_rows_dropped"] = int(zero.sum())
        if zero.any():  # undefined under cosine: drop them (recorded in meta.json)
            X = np.ascontiguousarray(X[~zero])
        meta["preprocessing"] = "raw train vectors; cosine metric (all-zero rows dropped)"
    else:
        meta["preprocessing"] = "raw train vectors"
    _save(name, X, None, meta, root)


def _synthetic(n: int) -> tuple[np.ndarray, np.ndarray]:
    p = SYNTH
    if not 1 <= n <= p["n_max"]:
        raise ValueError(f"synthetic n must be in [1, {p['n_max']}], got {n}.")
    C, k, d = p["n_components"], p["sub_dim"], p["d"]
    g = np.random.default_rng([p["seed"], 0])
    mu = (g.standard_normal((C, d)) * p["mean_scale"]).astype(np.float32)
    B = g.standard_normal((C, k, d)).astype(np.float32)
    X = np.empty((n, d), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    for c0 in range(0, n, _SYNTH_CHUNK):
        m = min(_SYNTH_CHUNK, p["n_max"] - c0)  # the chunk is always drawn whole: prefixes nest
        g = np.random.default_rng([p["seed"], 1, c0 // _SYNTH_CHUNK])
        lab = g.integers(0, C, size=m)
        z = g.standard_normal((m, k), dtype=np.float32)
        x = g.standard_normal((m, d), dtype=np.float32) * np.float32(p["noise"])
        for c in np.unique(lab):
            r = lab == c
            x[r] += mu[c] + z[r] @ B[c]
        take = min(m, n - c0)
        X[c0 : c0 + take] = x[:take]
        y[c0 : c0 + take] = lab[:take]
    return X, y


# ----------------------------------------------------------------------------- public

def ensure(name: str, root: Path = DATA_DIR) -> None:
    """Download and cache ``name`` if ``data/<name>/X.npy`` is missing."""
    if name == "synthetic" or (root / name / "X.npy").exists():
        return
    if name in ("mnist", "fashion_mnist"):
        _build_idx(name, root)
    elif name == "covtype":
        _build_covtype(root)
    elif name in SPECS and "hdf5" in SPECS[name]:
        _build_hdf5(name, root)
    else:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(SPECS)}")


def subset_indices(n_full: int, n: int) -> np.ndarray:
    """Rows used when only ``n`` of ``n_full`` real rows are wanted: a fixed random subset.

    The subsets are nested (the first ``n`` of one fixed permutation), kept in file order.
    """
    perm = np.random.default_rng(0).permutation(n_full)
    return np.sort(perm[:n])


def load(name: str, n: int | None = None, root: Path = DATA_DIR) -> Dataset:
    """Load (downloading on first use) dataset ``name``, optionally only ``n`` rows.

    Synthetic data uses nested prefixes; real data a nested fixed random subset.
    """
    if name not in SPECS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(SPECS)}")
    spec = SPECS[name]
    info: dict[str, Any] = {}
    if name == "synthetic":
        n_full = SYNTH["n_max"]
        X, y = _synthetic(n_full if n is None else n)
        info["generator"] = dict(SYNTH)
    else:
        ensure(name, root)
        X = np.load(root / name / "X.npy")
        y = np.load(root / name / "y.npy") if spec["labels"] else None
        _verify(name, X, y)
        n_full = X.shape[0]
        meta_path = root / name / "meta.json"
        if meta_path.exists():
            info["meta"] = json.loads(meta_path.read_text())
        if n is not None and n < n_full:
            idx = subset_indices(n_full, n)
            X = np.ascontiguousarray(X[idx])
            y = None if y is None else y[idx]
    if n is not None and X.shape[0] != n:
        raise ValueError(f"{name} has {X.shape[0]} rows, fewer than n={n}.")
    X = np.ascontiguousarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: non-finite values.")
    return Dataset(name, X, y, spec["metric"], n_full, info)


def main(argv: list[str]) -> int:
    names = argv or CORE
    status = 0
    for name in names:
        t = time.perf_counter()
        try:
            ds = load(name, n=100_000 if name == "synthetic" else None)
            lab = "none" if ds.y is None else f"{np.unique(ds.y).size} classes {np.bincount(ds.y - ds.y.min()).tolist()}"
            _log(f"OK {name}: shape {ds.X.shape} (full n {ds.n_full}) dtype {ds.X.dtype} "
                 f"C={ds.X.flags.c_contiguous} finite=True metric={ds.metric} "
                 f"min/mean/max {ds.X.min():.4g}/{ds.X.mean():.4g}/{ds.X.max():.4g} labels: {lab} "
                 f"meta: {json.dumps({k: v for k, v in ds.info.get('meta', {}).items() if k != 'md5'})} "
                 f"({time.perf_counter() - t:.1f} s)")
        except Exception as err:  # noqa: BLE001 - report every dataset, then fail
            status = 1
            _log(f"FAILED {name}: {err!r}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
