"""Run the eumap benchmark on one dataset (see benchmarks/PROTOCOL.md).

Every arm runs sequentially in this process, on this node. Each finished
(arm, seed) is appended as one JSON line to ``--out`` and flushed with fsync,
under a stable ``run_id``; a restarted run skips the run_ids already recorded
with ``status == "ok"``. A line cut short by a kill is ignored by the reader
and the next record starts on a fresh line.

Timers (``time.perf_counter``) wrap exactly the stage they name. Recall,
quality metrics and bookkeeping copies are always computed outside them.

Examples (from the repository root, inside a Slurm job)::

    python -I -u benchmarks/run_benchmark.py --dataset mnist
    python -I -u benchmarks/run_benchmark.py --dataset synthetic --n 1000000 --gt-only
    python -I -u benchmarks/run_benchmark.py --dataset deep10m --knn-only --seeds 0
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import socket
import sys
import threading
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import faiss  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import umap  # noqa: E402
import umap.umap_ as umap_mod  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.manifold import trustworthiness  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.utils import check_random_state  # noqa: E402
from threadpoolctl import threadpool_info  # noqa: E402

import eumap  # noqa: E402
from eumap import build_ivf_index, default_nlist, exact_knn, knn_recall, prepare, search_self, train_centroids  # noqa: E402

import datasets as dsets  # noqa: E402

K = 15  # n_neighbors everywhere (self included, as in UMAP)
K_GT = 16  # exact graph width: 15 non-self neighbours for NP@15
KNN_ARMS = ["nnd_modern", "nnd_legacy", "ivf_lloyd", "ivf_lloyd25", "ivf_emeans", "ivf_emeans2"]
E2E_ARMS = ["umap_e2e", "eumap_e2e", "eumap_lloyd_e2e", "layout_exact"]
ALL_ARMS = KNN_ARMS + E2E_ARMS
IVF_ARMS = {  # arm -> (trainer, trainer_kwds); "ALL_ROWS" is replaced by n (no subsampling)
    "ivf_lloyd": ("lloyd", {}),
    "ivf_lloyd25": ("lloyd", {"niter": 25, "max_points_per_centroid": "ALL_ROWS"}),
    "ivf_emeans": ("emeans", {}),
    "ivf_emeans2": ("emeans", {"nepochs": 2}),
}
RECALL_STOP = 0.999
NPROBE_MAX = 256
UMAP_COMMON = dict(n_neighbors=K, optimizer="adam", compatibility_layout=False, n_jobs=-1)
TW_ROWS = 10_000
TW_SEED = 12345
WARM_ROWS = 5_000
SAMPLED_QUERIES = 10_000
SAMPLED_SEED = 7
CV_SEED = 0


def log(msg: str) -> None:
    print(f"[bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- JSONL log

def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return repr(o)


class JsonlLog:
    """Append-only JSONL file that survives being killed at any point."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        self.records: list[dict[str, Any]] = []
        self.n_bad = 0
        if self.path.exists():
            for line in self.path.read_bytes().split(b"\n"):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    self.n_bad += 1  # a line cut short by a kill
                    continue
                self.records.append(rec)
                if rec.get("status") == "ok":
                    self.done.add(rec["run_id"])

    def write(self, rec: dict[str, Any]) -> None:
        line = (json.dumps(rec, default=_json_default, separators=(",", ":")) + "\n").encode()
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o664)
        try:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                line = b"\n" + line  # never glue onto a half-written line
            view = memoryview(line)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        if rec.get("status") == "ok":
            self.done.add(rec["run_id"])


# ----------------------------------------------------------------------------- memory

def _proc_status_kb(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


class RSSMonitor:
    """Peak resident memory per stage.

    A daemon thread samples VmRSS every ``interval`` s; in addition the kernel's
    exact high-water mark (VmHWM) is reset at the start of each stage through
    ``/proc/self/clear_refs`` where the kernel allows it. The stage peak is the
    larger of the two.
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_kb = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            v = _proc_status_kb("VmRSS")
            if v > self.peak_kb:
                self.peak_kb = v
            time.sleep(self.interval)

    @staticmethod
    def _reset_hwm() -> bool:
        try:
            with open("/proc/self/clear_refs", "w") as f:
                f.write("5")
            return True
        except OSError:
            return False

    @contextmanager
    def stage(self) -> Iterator[dict[str, Any]]:
        out: dict[str, Any] = {}
        self.peak_kb = start = _proc_status_kb("VmRSS")
        hwm_ok = self._reset_hwm()
        yield out
        sampled = max(self.peak_kb, _proc_status_kb("VmRSS"))
        hwm = _proc_status_kb("VmHWM") if hwm_ok else 0
        out["rss_start_gb"] = start * 1024 / 1e9
        out["peak_rss_gb"] = max(sampled, hwm) * 1024 / 1e9
        out["peak_rss_sampled_gb"] = sampled * 1024 / 1e9
        out["peak_rss_hwm_gb"] = hwm * 1024 / 1e9 if hwm_ok else None


MON: RSSMonitor


# ----------------------------------------------------------------------------- environment

def cpu_model() -> str:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def setup_threads() -> dict[str, Any]:
    n = len(os.sched_getaffinity(0))
    faiss.omp_set_num_threads(n)
    if numba.config.NUMBA_NUM_THREADS >= n:
        numba.set_num_threads(n)
    return thread_info()


def thread_info() -> dict[str, Any]:
    n = len(os.sched_getaffinity(0))
    blas = [{"api": i.get("internal_api"), "lib": os.path.basename(i.get("filepath", "")),
             "threads": i.get("num_threads")} for i in threadpool_info()]
    info = {
        "threads_affinity": n,
        "threads_faiss": faiss.omp_get_max_threads(),
        "threads_numba": numba.get_num_threads(),
        "threads_blas": blas,
        "env_threads": {v: os.environ.get(v) for v in
                        ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    info["thread_parity"] = (info["threads_faiss"] == n and info["threads_numba"] == n
                             and all(b["threads"] == n for b in blas))
    return info


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("eumap", "faiss-cpu", "umap-learn", "pynndescent", "numba", "numpy", "scipy", "scikit-learn"):
        try:
            out[pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = None
    return out


# ----------------------------------------------------------------------------- graph helpers

def self_first(I: np.ndarray, D: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copy of (I, D) with each row's own id in column 0 at distance 0 (bookkeeping, untimed)."""
    I, D = I.copy(), D.copy()
    k = I.shape[1]
    for r in np.flatnonzero(I[:, 0] != ids):
        keep = I[r] != ids[r]
        I[r] = np.concatenate(([ids[r]], I[r][keep]))[:k]
        D[r] = np.concatenate(([0.0], D[r][keep]))[:k]
    D[:, 0] = 0.0
    return I, D


def recall_stats(I: np.ndarray, D: np.ndarray, ctx: "Ctx") -> dict[str, Any]:
    """Recall of an n x K graph (or its sampled rows) against the exact graph."""
    if ctx.gt_rows is not None:
        I, D = I[ctx.gt_rows], D[ctx.gt_rows]
    G = np.ascontiguousarray(ctx.gt_I[:, :K])
    thr = ctx.gt_D[:, K - 1].astype(np.float64) * (1 + 1e-5)
    hits_tie = 0
    for s in range(0, I.shape[0], 65536):
        a, g = I[s : s + 65536, 1:], G[s : s + 65536, 1:]
        found = (a[:, :, None] == g[:, None, :]).any(axis=2) | (D[s : s + 65536, 1:] <= thr[s : s + 65536, None])
        hits_tie += int((found & (a >= 0)).sum())
    return {
        "recall": knn_recall(I, G),
        "recall_tie": hits_tie / (I.shape[0] * (K - 1)),
        "missing_frac": float((I < 0).mean()),
        "recall_sampled": ctx.gt_rows is not None,
    }


def np_at_k(E: np.ndarray, gt_I: np.ndarray, k: int = 15) -> float:
    """Neighbourhood preservation: overlap of the exact 2-D and high-dim k-NN (self excluded).

    The low-dimensional k-NN is exact, from a k-d tree in float64: brute force
    is O(n^2) and would take hours of an exclusive node at n = 3e6.
    """
    n = E.shape[0]
    E = np.ascontiguousarray(E, dtype=np.float64)
    tree = cKDTree(E)
    workers = len(os.sched_getaffinity(0))
    hits = 0
    for s in range(0, n, 65536):
        _, I2 = tree.query(E[s : s + 65536], k=k + 1, workers=workers)
        ids = np.arange(s, s + I2.shape[0])
        keep = I2 != ids[:, None]
        keep[keep.sum(axis=1) > k, -1] = False  # self not returned (ties): drop the last
        A = I2[keep].reshape(-1, k)
        B = gt_I[s : s + 65536, 1 : k + 1]
        hits += int((A[:, :, None] == B[:, None, :]).any(axis=2).sum())
    return hits / (n * k)


# ----------------------------------------------------------------------------- context

@dataclass
class Ctx:
    name: str
    n: int
    X: np.ndarray
    y: np.ndarray | None
    metric: str
    gt_I: np.ndarray  # (n or n_queries, K_GT), self first
    gt_D: np.ndarray
    gt_rows: np.ndarray | None  # sampled query rows, or None for the full graph
    tw_idx: np.ndarray | None = None
    quality: bool = True


def quality_metrics(E: np.ndarray, ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = {"embedding_finite": bool(np.isfinite(E).all())}
    if not ctx.quality or not out["embedding_finite"] or ctx.gt_rows is not None:
        return out
    t = time.perf_counter()
    out["np15"] = np_at_k(E, ctx.gt_I, 15)
    out["t_np15"] = time.perf_counter() - t
    t = time.perf_counter()
    idx = ctx.tw_idx
    out["trustworthiness"] = float(trustworthiness(ctx.X[idx], E[idx], n_neighbors=15, metric=ctx.metric))
    out["t_trustworthiness"] = time.perf_counter() - t
    if ctx.y is not None:
        t = time.perf_counter()
        cv = StratifiedKFold(5, shuffle=True, random_state=CV_SEED)
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=15, n_jobs=-1), E, ctx.y, cv=cv)
        out["knn_acc"] = float(scores.mean())
        out["knn_acc_folds"] = [float(s) for s in scores]
        out["t_knn_acc"] = time.perf_counter() - t
    return out


# ----------------------------------------------------------------------------- arms

def arm_nnd(ctx: Ctx, arm: str, seed: int) -> dict[str, Any]:
    legacy = arm == "nnd_legacy"
    rs = check_random_state(seed)
    angular = ctx.metric == "cosine"
    with MON.stage() as mem:
        t0 = time.perf_counter()
        I, D, index = umap_mod.nearest_neighbors(
            ctx.X, K, ctx.metric, {}, angular, rs, low_memory=True, use_pynndescent=True,
            n_jobs=-1, compatibility_layout=legacy,
        )
        t_knn = time.perf_counter() - t0
    del index
    n = ctx.X.shape[0]
    rec = {"t_knn": t_knn, "mem": {"knn": mem},
           "params": {"compatibility_layout": legacy, "n_iters": max(5, int(round(np.log2(n)))),
                      "n_trees": (min(64, 5 + int(round(n**0.5 / 20))) if legacy
                                  else max(4, min(8, 5 + int(round(n**0.5 / 20))))),
                      "max_candidates": 60 if legacy else max(20, min(60, int(1.5 * K)))}}
    Is, Ds = self_first(I, D, np.arange(n))
    rec["self_fixed_rows"] = int((I[:, 0] != np.arange(n)).sum())
    rec.update(recall_stats(Is, Ds, ctx))
    return rec


def arm_ivf(ctx: Ctx, arm: str, seed: int) -> dict[str, Any]:
    trainer, kw = IVF_ARMS[arm]
    n = ctx.X.shape[0]
    kw = {k: (n if v == "ALL_ROWS" else v) for k, v in kw.items()}
    nlist = default_nlist(n)
    mem: dict[str, Any] = {}
    with MON.stage() as mem["prepare"]:
        t0 = time.perf_counter()
        Xp = prepare(ctx.X, ctx.metric)
        t_prepare = time.perf_counter() - t0
    with MON.stage() as mem["train"]:
        t0 = time.perf_counter()
        C = train_centroids(Xp, nlist, trainer, spherical=ctx.metric == "cosine", seed=seed, **kw)
        t_train = time.perf_counter() - t0
    with MON.stage() as mem["add"]:
        t0 = time.perf_counter()
        index = build_ivf_index(Xp, C, ctx.metric)
        t_add = time.perf_counter() - t0
    # quantizer quality (untimed)
    _, A = index.quantizer.search(Xp, 1)
    A = A.ravel()
    sq = 0.0
    for s in range(0, n, 262144):
        diff = Xp[s : s + 262144] - C[A[s : s + 262144]]
        sq += float(np.einsum("ij,ij->", diff, diff, dtype=np.float64))
    sizes = np.bincount(A, minlength=nlist)
    rec: dict[str, Any] = {
        "params": {"trainer": trainer, "trainer_kwds": kw, "nlist": nlist, "spherical": ctx.metric == "cosine",
                   "nprobe_max": NPROBE_MAX, "recall_stop": RECALL_STOP},
        "t_prepare": t_prepare, "t_train": t_train, "t_add": t_add,
        "centroid_mse": sq / n,
        "imbalance": float(nlist * np.square(sizes, dtype=np.float64).sum() / float(n) ** 2),
        "max_list_over_mean": float(sizes.max() * nlist / n),
        "empty_lists": int((sizes == 0).sum()),
        "sweep": [],
    }
    del A
    nprobe = 1
    while nprobe <= min(nlist, NPROBE_MAX):
        with MON.stage() as m:
            t0 = time.perf_counter()
            I, D = search_self(index, Xp, K, nprobe, ctx.metric, repair=False)
            t_search = time.perf_counter() - t0
        point = {"nprobe": nprobe, "t_search": t_search, "peak_rss_gb": m["peak_rss_gb"]}
        point.update(recall_stats(I, D, ctx))
        rec["sweep"].append(point)
        log(f"    {arm} nprobe={nprobe:4d} recall={point['recall']:.4f} t_search={t_search:.2f}s")
        del I, D
        if point["recall"] >= RECALL_STOP:
            break
        nprobe *= 2
    last = rec["sweep"][-1]
    rec["sweep_end"] = ("recall" if last["recall"] >= RECALL_STOP
                        else "nlist" if 2 * last["nprobe"] > nlist and nlist <= NPROBE_MAX else "nprobe_max")
    rec["mem"] = mem
    return rec


class StageTimers:
    """Times UMAP's own kNN and fuzzy-simplicial-set calls by wrapping the module globals."""

    def __init__(self) -> None:
        self.t = {"nearest_neighbors": 0.0, "fuzzy_simplicial_set": 0.0}

    def _wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*a: Any, **k: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.t[name] += time.perf_counter() - t0
        return wrapper

    def __enter__(self) -> "StageTimers":
        self._orig = {n: getattr(umap_mod, n) for n in self.t}
        for n, fn in self._orig.items():
            setattr(umap_mod, n, self._wrap(n, fn))
        return self

    def __exit__(self, *exc: Any) -> None:
        for n, fn in self._orig.items():
            setattr(umap_mod, n, fn)


def _fit(model: Any, X: np.ndarray) -> tuple[float, dict[str, float], dict[str, Any], list[str]]:
    with warnings.catch_warnings(record=True) as caught, StageTimers() as st, MON.stage() as mem:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        model.fit(X)
        t_e2e = time.perf_counter() - t0
    msgs = sorted({str(w.message)[:200] for w in caught})
    return t_e2e, st.t, mem, msgs


def arm_e2e(ctx: Ctx, arm: str, seed: int) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    common = dict(UMAP_COMMON, metric=ctx.metric, random_state=seed)
    if arm == "umap_e2e":
        model = umap.UMAP(**common)
        t_e2e, st, mem, msgs = _fit(model, ctx.X)
        t_knn, umap_model = st["nearest_neighbors"], model
    elif arm in ("eumap_e2e", "eumap_lloyd_e2e"):
        trainer = "lloyd" if arm == "eumap_lloyd_e2e" else "emeans"
        common.pop("n_neighbors")
        model = eumap.EUMAP(n_neighbors=K, exact_below=0, trainer=trainer, **common)
        t_e2e, st, mem, msgs = _fit(model, ctx.X)
        t_knn, umap_model = model.knn_info_["t_total"], model.umap_
        rec["knn_info"] = model.knn_info_
    elif arm == "layout_exact":
        I = np.ascontiguousarray(ctx.gt_I[:, :K]).copy()
        D = np.ascontiguousarray(ctx.gt_D[:, :K]).copy()
        model = umap.UMAP(precomputed_knn=(I, D), **common)
        t_e2e, st, mem, msgs = _fit(model, ctx.X)
        t_knn, umap_model = 0.0, model
    else:
        raise ValueError(arm)
    rec.update({
        "t_e2e": t_e2e, "t_knn": t_knn, "t_graph": st["fuzzy_simplicial_set"],
        "t_layout": t_e2e - t_knn - st["fuzzy_simplicial_set"],
        "mem": {"e2e": mem}, "warnings": msgs,
        "params": {k: v for k, v in common.items()},
    })
    # graph recall of the kNN graph UMAP actually used (untimed)
    n = ctx.X.shape[0]
    gI = np.asarray(umap_model._knn_indices)
    gD = np.asarray(umap_model._knn_dists)
    if gI.shape == (n, K):
        Is, Ds = self_first(gI, gD, np.arange(n))
        rec.update({f"graph_{k}": v for k, v in recall_stats(Is, Ds, ctx).items()})
    E = np.asarray(umap_model.embedding_)
    rec.update(quality_metrics(E, ctx))
    return rec


def run_arm(ctx: Ctx, arm: str, seed: int) -> dict[str, Any]:
    if arm in ("nnd_modern", "nnd_legacy"):
        return arm_nnd(ctx, arm, seed)
    if arm in IVF_ARMS:
        return arm_ivf(ctx, arm, seed)
    return arm_e2e(ctx, arm, seed)


# ----------------------------------------------------------------------------- ground truth

def gt_path(root: Path, name: str, n: int, n_full: int, sampled: bool) -> Path:
    if sampled:
        return root / name / (f"gt_sampled_q{SAMPLED_QUERIES}_k{K_GT}.npz" if n == n_full
                              else f"gt_sampled_n{n}_q{SAMPLED_QUERIES}_k{K_GT}.npz")
    return root / name / (f"gt_k{K_GT}.npz" if n == n_full else f"gt_n{n}_k{K_GT}.npz")


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def data_fingerprint(ds: dsets.Dataset, checksum: float) -> dict[str, Any]:
    """What a cached ground truth must have been computed from: shape, metric, generator and a content hash."""
    h = hashlib.blake2b(digest_size=16)
    step = max(1, (64 << 20) // max(1, ds.X.shape[1] * ds.X.itemsize))  # about 64 MB per update
    for s in range(0, ds.n, step):
        h.update(np.ascontiguousarray(ds.X[s : s + step]).data)
    return {"n": int(ds.n), "d": int(ds.X.shape[1]), "metric": ds.metric, "k": K_GT, "sampled": not ds.full_gt,
            "sampled_queries": SAMPLED_QUERIES if not ds.full_gt else None,
            "sampled_seed": SAMPLED_SEED if not ds.full_gt else None,
            "generator": ds.info.get("generator"), "checksum_f64": checksum, "blake2b": h.hexdigest()}


def _load_gt(path: Path, ds: dsets.Dataset, fp: dict[str, Any]) -> tuple[Any, ...] | None:
    """The cached graph if it was computed from exactly this data and has the right shape, else None."""
    sampled = not ds.full_gt
    try:
        z = np.load(path)
        I, D = z["I"], z["D"]
        rows = z["rows"] if sampled else None
        meta = json.loads(str(z["meta"]))
    except Exception as err:  # noqa: BLE001 - unreadable cache: recompute
        log(f"WARNING: ground truth {path} is unreadable ({err!r}); recomputing")
        return None
    old = meta.get("data")
    if old != json.loads(json.dumps(fp)):  # (JSON round trip so tuples/lists compare equal)
        diff = sorted(k for k in set(fp) | set(old or {}) if (old or {}).get(k) != fp.get(k))
        log(f"WARNING: ground truth {path} was computed from other data (differs in {diff or 'no fingerprint'}); "
            "recomputing")
        return None
    n_rows = SAMPLED_QUERIES if sampled else ds.n
    ok = (I.shape == (n_rows, K_GT) and D.shape == (n_rows, K_GT) and int(I.min()) >= 0 and int(I.max()) < ds.n
          and (not sampled or (rows is not None and rows.shape == (n_rows,) and np.array_equal(I[:, 0], rows)))
          and (sampled or np.array_equal(I[:, 0], np.arange(ds.n))))
    if not ok:
        log(f"WARNING: ground truth {path} has the wrong shape or content ({I.shape}); recomputing")
        return None
    return I, D, rows, meta


def sampled_exact(Xp: np.ndarray, rows: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Exact K_GT neighbours of the rows ``rows`` of prepared data, self first.

    Faiss brute force finds the ids; their distances are then recomputed directly
    (not in the BLAS form |x|^2 + |y|^2 - 2<x, y>) and each row re-sorted, as
    eumap.exact_knn does for the full graph.
    """
    mt = faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2
    _, I = faiss.knn(Xp[rows], Xp, K_GT, metric=mt)
    diff = Xp[I] - Xp[rows][:, None, :]
    sq = np.einsum("ijd,ijd->ij", diff, diff, dtype=np.float64)
    D = (0.5 * sq) if metric == "cosine" else np.sqrt(sq)  # unit vectors: 1 - cos = |x - y|^2 / 2
    order = np.argsort(D, axis=1, kind="stable")
    I = np.take_along_axis(I, order, axis=1)
    D = np.take_along_axis(D, order, axis=1).astype(np.float32)
    return self_first(I, D, rows)


def ground_truth(ds: dsets.Dataset, root: Path, env: dict[str, Any], jlog: JsonlLog,
                 base: dict[str, Any], fp: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    sampled = not ds.full_gt
    path = gt_path(root, ds.name, ds.n, ds.n_full, sampled)
    computed = False
    cached = _load_gt(path, ds, fp) if path.exists() else None
    if cached is not None:
        I, D, rows, meta = cached
        log(f"ground truth loaded from {path} (computed in {meta['t_exact']:.1f} s on {meta['hostname']}; "
            "data fingerprint matches)")
    else:
        log(f"computing exact {'sampled ' if sampled else ''}ground truth k={K_GT} for n={ds.n} ...")
        with MON.stage() as mem:
            if sampled:
                rows = np.sort(np.random.default_rng(SAMPLED_SEED).choice(ds.n, SAMPLED_QUERIES, replace=False))
                t0 = time.perf_counter()
                Xp = prepare(ds.X, ds.metric)
                I, D = sampled_exact(Xp, rows, ds.metric)
                t_exact = time.perf_counter() - t0
                del Xp
            else:
                rows = None
                t0 = time.perf_counter()
                I, D = exact_knn(ds.X, K_GT, ds.metric)
                t_exact = time.perf_counter() - t0
        meta = {"t_exact": t_exact, "peak_rss_gb": mem["peak_rss_gb"], "data": fp, **{k: env[k] for k in
                ("hostname", "cpu_model", "job_id", "threads_affinity", "threads_faiss")}}
        arrays = {"I": I, "D": D, "meta": json.dumps(meta)}
        if sampled:
            arrays["rows"] = rows
        _save_npz(path, **arrays)
        computed = True
        log(f"ground truth done in {t_exact:.1f} s -> {path}")
    run_id = f"{ds.name}|n={ds.n}|k={K_GT}|arm=exact"
    if run_id not in jlog.done:
        jlog.write({**base, "run_id": run_id, "status": "ok", "arm": "exact", "seed": None,
                    "t_knn": meta["t_exact"], "gt_path": str(path), "gt_computed_here": computed,
                    "gt_meta": meta, "recall": 1.0, "recall_sampled": sampled,
                    "mem": {"knn": {"peak_rss_gb": meta.get("peak_rss_gb")}}})
    return I, D, rows


# ----------------------------------------------------------------------------- main

def rotated(arms: list[str], i: int, n_seeds: int) -> list[str]:
    shift = (i * len(arms)) // max(n_seeds, 1) % max(len(arms), 1)
    return arms[shift:] + arms[:shift]


def warmup(ds: dsets.Dataset, arms: list[str], ctx_full: Ctx) -> float:
    """Run every selected arm once on the first WARM_ROWS rows (JIT compilation, thread pools)."""
    t0 = time.perf_counter()
    Xw = np.ascontiguousarray(ds.X[:WARM_ROWS])
    gI, gD = exact_knn(Xw, K_GT, ds.metric)
    ctx = Ctx(ds.name, Xw.shape[0], Xw, None if ds.y is None else ds.y[:WARM_ROWS], ds.metric,
              gI, gD, None, np.arange(Xw.shape[0]), quality=False)
    for arm in arms:
        t = time.perf_counter()
        run_arm(ctx, arm, 0)
        log(f"  warm-up {arm}: {time.perf_counter() - t:.1f} s")
    # the quality code is untimed, but warm it too so its first call is not an outlier
    quality_metrics(np.ascontiguousarray(Xw[:, :2]), Ctx(ds.name, Xw.shape[0], Xw, ctx.y, ds.metric, gI, gD,
                                                         None, np.arange(min(2000, Xw.shape[0]))))
    return time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(dsets.SPECS))
    p.add_argument("--n", type=int, nargs="+", help="sizes (synthetic: prefixes; real: fixed nested subsets)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", nargs="+", default=None, choices=ALL_ARMS + ["exact"],
                   help="default: all (kNN arms only with --knn-only); 'exact' is always computed or loaded")
    p.add_argument("--knn-only", action="store_true", help="no layout arms (required for datasets without a full graph)")
    p.add_argument("--gt-only", action="store_true", help="only compute and cache the exact graph")
    p.add_argument("--out", type=Path, default=None, help="JSONL file (default results/raw/<dataset>.jsonl)")
    p.add_argument("--data-dir", type=Path, default=dsets.DATA_DIR)
    args = p.parse_args(argv)

    global MON
    MON = RSSMonitor()
    out = args.out or Path("results/raw") / f"{args.dataset}.jsonl"
    jlog = JsonlLog(out)
    knn_only = args.knn_only or not dsets.SPECS[args.dataset].get("full_gt", True)
    arms = [a for a in (args.arms or ALL_ARMS) if a != "exact"]
    if knn_only:
        arms = [a for a in arms if a in KNN_ARMS]
    sizes = args.n or (dsets.SYNTH_SIZES if args.dataset == "synthetic" else [None])

    env = {"hostname": socket.gethostname(), "cpu_model": cpu_model(),
           "job_id": os.environ.get("SLURM_JOB_ID"), "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
           **setup_threads()}
    env["versions"] = versions()
    log(f"out={out} ({len(jlog.done)} finished records, {jlog.n_bad} unreadable lines) arms={arms} "
        f"seeds={args.seeds} sizes={sizes} knn_only={knn_only}")
    log(f"host={env['hostname']} cpu='{env['cpu_model']}' threads: affinity={env['threads_affinity']} "
        f"faiss={env['threads_faiss']} numba={env['threads_numba']} blas={env['threads_blas']} "
        f"parity={env['thread_parity']}")
    if not env["thread_parity"]:
        log("WARNING: thread counts differ from the affinity mask; recorded in every record.")

    t = time.perf_counter()
    dsets.ensure(args.dataset, args.data_dir)
    t_download = time.perf_counter() - t

    n_fail = 0
    for n_req in sizes:
        t = time.perf_counter()
        ds = dsets.load(args.dataset, n_req, args.data_dir)
        touch = float(ds.X.sum(dtype=np.float64))  # first touch of every page, before any timer
        t_load = time.perf_counter() - t
        t = time.perf_counter()
        fp = data_fingerprint(ds, touch)
        log(f"dataset {ds.name}: n={ds.n} d={ds.X.shape[1]} metric={ds.metric} labels={ds.y is not None} "
            f"load {t_load:.1f} s (checksum {touch:.6g}, blake2b {fp['blake2b']} in {time.perf_counter() - t:.1f} s)")
        base = {"dataset": ds.name, "n": ds.n, "d": ds.X.shape[1], "n_full": ds.n_full, "metric": ds.metric,
                "has_labels": ds.y is not None, "k": K, "t_load": t_load, "t_download": t_download,
                "knn_only": knn_only, "data_fingerprint": fp["blake2b"], **env}

        gI, gD, rows = ground_truth(ds, args.data_dir, env, jlog, base, fp)
        if args.gt_only:
            continue
        tw_idx = (np.sort(np.random.default_rng(TW_SEED).choice(ds.n, TW_ROWS, replace=False))
                  if ds.n > TW_ROWS else np.arange(ds.n))
        ctx = Ctx(ds.name, ds.n, ds.X, ds.y, ds.metric, gI, gD, rows, tw_idx, quality=not knn_only)

        def rid(arm: str, seed: int) -> str:
            return f"{ds.name}|n={ds.n}|k={K}|arm={arm}|seed={seed}"

        todo = [(s, a) for s in args.seeds for a in arms if rid(a, s) not in jlog.done]
        if not todo:
            log(f"n={ds.n}: all {len(args.seeds) * len(arms)} records already present, skipping")
            continue
        log(f"n={ds.n}: {len(todo)} records to run; warming up on {WARM_ROWS} rows ...")
        t_warmup = warmup(ds, arms, ctx)
        log(f"warm-up {t_warmup:.1f} s")
        for i, seed in enumerate(args.seeds):
            order = rotated(arms, i, len(args.seeds))
            for pos, arm in enumerate(order):
                run_id = rid(arm, seed)
                if run_id in jlog.done:
                    continue
                gc.collect()
                log(f"run {run_id} (position {pos})")
                rec = {**base, "run_id": run_id, "arm": arm, "seed": seed, "order": order, "order_pos": pos,
                       "t_warmup": t_warmup, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
                t = time.perf_counter()
                try:
                    rec.update(run_arm(ctx, arm, seed))
                    rec["status"] = "ok"
                except Exception as err:  # noqa: BLE001 - record and continue with the next arm
                    n_fail += 1
                    rec.update(status="error", error=repr(err), traceback=traceback.format_exc()[-3000:])
                    log(f"ERROR in {run_id}: {err!r}")
                rec["t_record"] = time.perf_counter() - t
                rec["threads_after"] = thread_info()
                jlog.write(rec)
                summary = {k: rec.get(k) for k in ("t_knn", "t_train", "t_e2e", "recall", "graph_recall",
                                                   "np15", "trustworthiness", "knn_acc") if rec.get(k) is not None}
                log(f"  done {arm} seed={seed} in {rec['t_record']:.1f} s: "
                    + ", ".join(f"{k}={v:.4g}" for k, v in summary.items()))
    log(f"finished ({n_fail} failed records)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
