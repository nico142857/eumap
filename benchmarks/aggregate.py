"""Aggregate benchmark records into tables, figures and the pre-registered verdicts.

Reads every ``*.jsonl`` in ``--raw`` (default ``results/raw``), ignoring unreadable
(half-written) lines and keeping the last ``ok`` record per ``run_id``. Writes
``<out>/summary.csv``, ``<out>/RESULTS.md`` and ``<out>/figures/*.png`` and prints
one verdict line per criterion and dataset. Works on partial results: anything
that cannot be evaluated yet says so.

    python -I benchmarks/aggregate.py [--raw results/raw] [--out results]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

NND = ["nnd_modern", "nnd_legacy"]
IVF = ["ivf_lloyd", "ivf_lloyd25", "ivf_emeans", "ivf_emeans2"]
E2E = ["umap_e2e", "eumap_e2e", "eumap_lloyd_e2e", "layout_exact"]
QUALITY = ["np15", "trustworthiness", "knn_acc"]
RECALL_CAP = 0.999
NPROBE_MAX = 256  # sweep limit of run_benchmark.py (used when a record does not store it)
EVAL_SET = [("covtype", None), ("sift1m", None), ("glove100", None), ("synthetic", 1_000_000), ("gist1m", None)]

# Fixed arm -> colour (categorical slots of the reference palette) and marker; never by rank.
STYLE = {
    "nnd_modern": ("#2a78d6", "o"), "nnd_legacy": ("#6250d6", "s"),
    "ivf_lloyd": ("#eb6834", "^"), "ivf_lloyd25": ("#eda100", "v"),
    "ivf_emeans": ("#1baf7a", "D"), "ivf_emeans2": ("#008300", "P"),
    "exact": ("#52514e", "X"),
    "umap_e2e": ("#2a78d6", "o"), "eumap_e2e": ("#1baf7a", "D"),
    "eumap_lloyd_e2e": ("#eb6834", "^"), "layout_exact": ("#52514e", "X"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------- loading

def load(raw: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    ok: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    bad = 0
    for path in sorted(raw.glob("*.jsonl")):
        for line in path.read_bytes().split(b"\n"):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            rec["_file"] = path.name
            if rec.get("status") == "ok":
                ok[rec["run_id"]] = rec
            else:
                errors.append(rec)
    errors = [e for e in errors if e.get("run_id") not in ok]  # later retries succeeded
    return ok, errors, bad


def key_of(r: dict[str, Any]) -> tuple[str, int]:
    return (r["dataset"], int(r["n"]))


def fmt_n(n: int) -> str:
    """1000000 -> '1e6', 30000 -> '3e4', 70000 -> '7e4', 581012 -> '581012'."""
    m, e = f"{n:e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}e{int(e)}" if "." not in m else str(n)


def label(key: tuple[str, int], full: dict[tuple[str, int], int]) -> str:
    ds, n = key
    return f"{ds} n={fmt_n(n)}" if ds == "synthetic" or n != full.get(key, n) else ds


def med(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else None


def rng_(xs: list[float]) -> tuple[float, float] | None:
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if xs else None


def f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        if math.isinf(x):
            return "inf"
        return f"{x:.{nd}f}" if abs(x) < 1000 else f"{x:.0f}"
    return str(x)


def fr(xs: list[float], nd: int = 3) -> str:
    m, r = med(xs), rng_(xs)
    if m is None:
        return "–"
    return f"{f(m, nd)} [{f(r[0], nd)}–{f(r[1], nd)}]" if len(xs) > 1 else f(m, nd)


# ----------------------------------------------------------------------------- analysis

class Group:
    """All ok records of one (dataset, n), restricted to one host, plus its unresolved errors.

    Times are only compared within one node (PROTOCOL.md, "Hardware and software"): if the
    records of a (dataset, n) come from several hosts (a chained part-2 job or a manual
    resubmission that landed elsewhere), only the host with the most records is used for
    every table and verdict, and the others are listed in ``excluded``.
    """

    def __init__(self, key: tuple[str, int], recs: list[dict[str, Any]], errors: list[dict[str, Any]] = ()):
        self.key = key
        self.all_recs = recs
        hosts = Counter(r.get("hostname") for r in recs if r["arm"] != "exact")
        # most records first; ties go to the host name that sorts first (deterministic)
        self.host = min(hosts, key=lambda h: (-hosts[h], str(h))) if hosts else None
        self.excluded = [r for r in recs if r["arm"] != "exact" and r.get("hostname") != self.host]
        self.recs = [r for r in recs if r["arm"] == "exact" or r.get("hostname") == self.host]
        self.by_arm: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        self.exact: dict[str, Any] | None = None
        for r in self.recs:
            if r["arm"] == "exact":
                self.exact = r
            else:
                self.by_arm[r["arm"]][r["seed"]] = r
        self.errors: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for e in errors:
            if e.get("arm") and e.get("seed") is not None:
                self.errors[e["arm"]][e["seed"]] = e
        self.has_labels = any(r.get("has_labels") for r in recs)

    def vals(self, arm: str, field: str) -> list[float]:
        return [r.get(field) for _, r in sorted(self.by_arm.get(arm, {}).items()) if r.get(field) is not None]

    def faster_nnd(self) -> str | None:
        cands = [(med(self.vals(a, "t_knn")), a) for a in NND if self.vals(a, "t_knn")]
        return min(cands)[1] if cands else None

    def matched(self, ivf: str, nnd: str) -> list[dict[str, Any]]:
        """Per seed: IVF operating point with the smallest nprobe reaching NN-descent's recall.

        Pre-registered fields: ``ratio`` (inf when unmatched). Informational fields:
        ``ratio_interp`` (search time interpolated linearly in recall between the two swept
        nprobes that bracket the target, so the power-of-two grid's overshoot is visible),
        ``unmatched`` ("cap": the sweep hit nprobe <= 256; "nlist": every list was probed;
        "error": the IVF arm raised) and ``lb_ratio`` (for "cap"/"nlist": the IVF cost at
        the last swept nprobe over the NN-descent time, a lower bound on the ratio a longer
        sweep could have given, because search time grows with nprobe).
        """
        out = []
        for s, rn in sorted(self.by_arm.get(nnd, {}).items()):
            target = min(rn["recall"], RECALL_CAP)
            base = {"seed": s, "r_nnd": rn["recall"], "target": target, "capped": rn["recall"] > RECALL_CAP,
                    "t_nnd": rn["t_knn"]}
            ri = self.by_arm.get(ivf, {}).get(s)
            if ri is None:
                if s in self.errors.get(ivf, {}):  # the IVF arm failed on this seed: it cannot match
                    out.append({**base, "nprobe": None, "recall_ivf": None, "t_search": math.inf,
                                "t_ivf": math.inf, "ratio": math.inf, "t_ivf_interp": math.inf,
                                "ratio_interp": math.inf, "unmatched": "error", "lb_ratio": None})
                continue
            sweep = sorted(ri["sweep"], key=lambda p: p["nprobe"])
            fixed = ri["t_train"] + ri["t_add"]
            i = next((j for j, p in enumerate(sweep) if p["recall"] >= target), None)
            if i is None:
                last = sweep[-1]
                nlist = ri.get("params", {}).get("nlist") or math.inf
                cap = ri.get("params", {}).get("nprobe_max", NPROBE_MAX)
                why = "nlist" if 2 * last["nprobe"] > nlist and nlist <= cap else "cap"
                out.append({**base, "nprobe": None, "recall_ivf": last["recall"], "t_search": math.inf,
                            "t_ivf": math.inf, "ratio": math.inf, "t_ivf_interp": math.inf,
                            "ratio_interp": math.inf, "unmatched": why, "last_nprobe": last["nprobe"],
                            "lb_ratio": (fixed + last["t_search"]) / rn["t_knn"]})
                continue
            pt = sweep[i]
            t_int = pt["t_search"]
            if i > 0 and pt["recall"] > sweep[i - 1]["recall"]:
                lo = sweep[i - 1]
                w = (target - lo["recall"]) / (pt["recall"] - lo["recall"])
                t_int = lo["t_search"] + w * (pt["t_search"] - lo["t_search"])
            out.append({**base, "nprobe": pt["nprobe"], "recall_ivf": pt["recall"], "t_search": pt["t_search"],
                        "t_ivf": fixed + pt["t_search"], "ratio": (fixed + pt["t_search"]) / rn["t_knn"],
                        "t_ivf_interp": fixed + t_int, "ratio_interp": (fixed + t_int) / rn["t_knn"],
                        "unmatched": None, "lb_ratio": None})
        return out

    def ratio(self, ivf: str, nnd: str | None = None) -> tuple[float | None, list[dict[str, Any]]]:
        nnd = nnd or self.faster_nnd()
        if nnd is None:
            return None, []
        m = self.matched(ivf, nnd)
        return (med([x["ratio"] for x in m]) if m else None), m


def optimistic(m: list[dict[str, Any]]) -> float | None:
    """Median ratio with each cap/nlist-unmatched seed replaced by its lower bound (informational).

    If even this optimistic median fails a threshold, the nprobe cap did not decide the verdict.
    """
    xs = [x["lb_ratio"] if x["unmatched"] in ("cap", "nlist") else x["ratio"] for x in m]
    return med(xs) if xs else None


def cap_note(m: list[dict[str, Any]], thr: float, strict: bool = False) -> str:
    """Says whether the unmatched seeds (nprobe cap) could have changed a ratio verdict."""
    um = [x for x in m if x["unmatched"]]
    if not um:
        return ""
    note = "; unmatched seeds: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(x["unmatched"] for x in um).items()))
    if not any(x["unmatched"] in ("cap", "nlist") for x in um):
        return note

    def passes(r: float | None) -> bool:
        return r is not None and (r < thr if strict else r <= thr)

    actual, opt = med([x["ratio"] for x in m]), optimistic(m)
    if passes(actual):
        return note
    if passes(opt):
        return (note + f"; DECIDED BY THE NPROBE CAP: with the end-of-sweep cost as a lower bound for those "
                f"seeds the median would be {f(opt)}, which would pass")
    return (note + f"; not decided by the nprobe cap: even with the end-of-sweep cost as a lower bound the "
            f"median is >= {f(opt)}")


def crit_a(g: Group, ivf: str = "ivf_emeans") -> tuple[bool | None, str]:
    nnd = g.faster_nnd()
    r, m = g.ratio(ivf, nnd)
    if r is None:
        return None, f"(a) {ivf} vs faster NN-descent: insufficient data"
    rs = [x["ratio"] for x in m]
    ri = med([x["ratio_interp"] for x in m])
    return r <= 0.8, (f"(a) {ivf} kNN time at matched recall / {nnd} = {f(r)} "
                      f"[{f(min(rs))}–{f(max(rs))}] over {len(rs)} seeds (needs <= 0.8)"
                      f"{cap_note(m, 0.8)}; informational: {f(ri)} with search time interpolated between swept nprobes")


def _arm_errors(g: Group, arm: str) -> str:
    errs = g.errors.get(arm, {})
    if not errs:
        return ""
    first = next(iter(errs.values())).get("error", "")[:120]
    return f"seed(s) {sorted(errs)} raised an error ({first})"


def crit_b(g: Group, arm: str = "eumap_e2e") -> tuple[bool | None, str]:
    err = _arm_errors(g, arm)
    if err:  # a fit that fails is not faster (PROTOCOL.md: errors count as failures)
        return False, f"(b) {arm} / umap_e2e: FAIL, {err}"
    te, tu = med(g.vals(arm, "t_e2e")), med(g.vals("umap_e2e", "t_e2e"))
    if te is None or tu is None:
        ref = _arm_errors(g, "umap_e2e")
        return None, f"(b) {arm} / umap_e2e: insufficient data{f' (umap_e2e {ref})' if ref else ''}"
    return te / tu <= 0.85, (f"(b) {arm} / umap_e2e end-to-end = {f(te / tu)} "
                             f"({f(te, 1)} s vs {f(tu, 1)} s, medians) (needs <= 0.85)")


def crit_c(g: Group, arm: str = "eumap_e2e") -> tuple[bool | None, str]:
    # A non-finite embedding or a fit that raised is a quality failure, not missing data.
    broken = sorted(s for s, r in g.by_arm.get(arm, {}).items() if r.get("embedding_finite") is False)
    err = _arm_errors(g, arm)
    if broken or err:
        why = "; ".join(x for x in (f"non-finite embedding on seed(s) {broken}" if broken else "", err) if x)
        return False, f"(c) {arm} quality: FAIL, {why}"
    parts, ok_all, any_data = [], True, False
    for q in QUALITY:
        if q == "knn_acc" and not g.has_labels:
            continue
        e, u = g.vals(arm, q), g.vals("umap_e2e", q)
        if not e or not u:
            parts.append(f"{q}: missing")
            ok_all = False if any_data else ok_all
            continue
        any_data = True
        me, mu, lo = med(e), med(u), min(u)
        ok = me >= lo or me >= mu - 0.005
        ok_all &= ok
        rel = f" ({100 * (me - mu) / mu:+.1f}% of the umap median)" if mu else ""
        parts.append(f"{q} {f(me, 4)} vs umap {f(mu, 4)} [{f(lo, 4)}–{f(max(u), 4)}]{rel} {'ok' if ok else 'WORSE'}")
    ref_broken = sorted(s for s, r in g.by_arm.get("umap_e2e", {}).items() if r.get("embedding_finite") is False)
    if ref_broken:
        parts.append(f"note: umap_e2e non-finite on seed(s) {ref_broken}, its range uses the other seeds")
    if not any_data or any(p.endswith("missing") for p in parts):
        return None, f"(c) {arm} quality: insufficient data ({'; '.join(parts)})"
    return ok_all, f"(c) {arm} quality: " + "; ".join(parts)


def emeans_contrib(g: Group) -> tuple[bool | None, str, float | None]:
    tl = med(g.vals("ivf_lloyd", "t_train"))
    if tl is None:
        return None, "e-means contributes: no ivf_lloyd records", None
    nnd = g.faster_nnd()
    _, ml = g.ratio("ivf_lloyd", nnd)
    lloyd_search_max = max((x["t_search"] for x in ml), default=None)
    train_parts, train_ok = [], False
    for arm in ("ivf_emeans", "ivf_emeans2"):
        te = med(g.vals(arm, "t_train"))
        if te is None:
            continue
        _, me = g.ratio(arm, nnd)
        se = med([x["t_search"] for x in me]) if me else None
        # an unmatched operating point (inf) cannot show "equal or better" search cost
        ok = (te <= 0.7 * tl and se is not None and lloyd_search_max is not None
              and math.isfinite(se) and se <= lloyd_search_max)
        train_ok |= ok
        train_parts.append(f"{arm} t_train/lloyd {f(te / tl)} (<= 0.7), matched search {f(se, 2)} s vs lloyd max "
                           f"{f(lloyd_search_max, 2)} s -> {'ok' if ok else 'no'}")
    te2e, tl2e = med(g.vals("eumap_e2e", "t_e2e")), med(g.vals("eumap_lloyd_e2e", "t_e2e"))
    gain = (tl2e - te2e) / tl2e if te2e is not None and tl2e else None
    if not train_parts or gain is None:
        return None, "e-means contributes: insufficient data (" + "; ".join(train_parts) + ")", gain
    e2e_ok = te2e <= 0.95 * tl2e
    return (train_ok and e2e_ok, "e-means contributes: " + "; ".join(train_parts)
            + f"; eumap_e2e / eumap_lloyd_e2e = {f(te2e / tl2e)} (needs <= 0.95)", gain)


def evaluate(g: Group) -> dict[str, Any]:
    a, ta = crit_a(g)
    b, tb = crit_b(g)
    c, tc = crit_c(g)
    al, tal = crit_a(g, "ivf_lloyd")
    bl, tbl = crit_b(g, "eumap_lloyd_e2e")
    cl, tcl = crit_c(g, "eumap_lloyd_e2e")
    ec, tec, gain = emeans_contrib(g)
    worth = None if None in (a, b, c) else (a and b and c)
    worth_l = None if None in (al, bl, cl) else (al and bl and cl)
    ivf_helps = None
    if gain is not None and (worth is not None or worth_l is not None):
        ivf_helps = bool((worth or worth_l) and gain < 0.05)
    beats, beats_opt = {}, {}
    for arm in IVF:
        r, m = g.ratio(arm)
        if r is not None:
            beats[arm] = r
            beats_opt[arm] = optimistic(m)
    return {"a": (a, ta), "b": (b, tb), "c": (c, tc), "a_lloyd": (al, tal), "b_lloyd": (bl, tbl),
            "c_lloyd": (cl, tcl), "worth": worth, "worth_lloyd": worth_l, "emeans": (ec, tec), "gain": gain,
            "ivf_helps_emeans_not": ivf_helps, "beats": beats, "beats_opt": beats_opt}


def yn(x: bool | None) -> str:
    return "n/a" if x is None else ("YES" if x else "no")


# ----------------------------------------------------------------------------- outputs

def summary_frame(recs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    base_cols = ("dataset", "n", "d", "metric", "arm", "seed", "order_pos", "hostname", "job_id", "cpu_model",
                 "threads_affinity", "threads_faiss", "threads_numba", "thread_parity", "t_load", "t_warmup")
    for r in recs:
        base = {c: r.get(c) for c in base_cols}
        mem = r.get("mem", {})
        peak = max((v.get("peak_rss_gb") or 0 for v in mem.values()), default=None)
        if r["arm"] in IVF:
            for p in r["sweep"]:
                rows.append({**base, "nprobe": p["nprobe"], "recall": p["recall"], "recall_tie": p["recall_tie"],
                             "missing_frac": p["missing_frac"], "t_prepare": r["t_prepare"], "t_train": r["t_train"],
                             "t_add": r["t_add"], "t_search": p["t_search"],
                             "t_knn": r["t_train"] + r["t_add"] + p["t_search"], "nlist": r["params"]["nlist"],
                             "centroid_mse": r["centroid_mse"], "imbalance": r["imbalance"],
                             "peak_rss_gb": max(peak or 0, p.get("peak_rss_gb") or 0)})
        else:
            row = {**base, "peak_rss_gb": peak}
            for c in ("recall", "recall_tie", "missing_frac", "t_knn", "t_e2e", "t_graph", "t_layout",
                      "graph_recall", "np15", "trustworthiness", "knn_acc"):
                row[c] = r.get(c)
            if r.get("knn_info"):
                row["nprobe"] = r["knn_info"].get("nprobe")
                row["nlist"] = r["knn_info"].get("nlist")
            rows.append(row)
    return pd.DataFrame(rows)


def _ax_style(ax: Any) -> None:
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def fig_recall_time(g: Group, lab: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm in IVF:
        recs = g.by_arm.get(arm, {})
        if not recs:
            continue
        pts: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for r in recs.values():
            for p in r["sweep"]:
                pts[p["nprobe"]].append((r["t_train"] + r["t_add"] + p["t_search"], p["recall"]))
        xs = [med([t for t, _ in pts[k]]) for k in sorted(pts)]
        ys = [med([rc for _, rc in pts[k]]) for k in sorted(pts)]
        c, mk = STYLE[arm]
        ax.plot(xs, ys, color=c, marker=mk, ms=5, lw=2, label=arm)
    for arm in NND:
        recs = g.by_arm.get(arm, {})
        if not recs:
            continue
        c, mk = STYLE[arm]
        xs = [r["t_knn"] for r in recs.values()]
        ys = [r["recall"] for r in recs.values()]
        ax.scatter(xs, ys, color=c, marker=mk, s=60, label=arm, zorder=3, edgecolor="white", linewidth=1.5)
    if g.exact:
        c, mk = STYLE["exact"]
        if g.exact.get("recall_sampled"):  # brute force for 10k queries only: not a full graph
            ax.scatter([g.exact["t_knn"]], [1.0], facecolors="none", edgecolors=c, marker="o", s=70, zorder=3,
                       label="exact, 10k sampled queries only (time not comparable)")
        else:
            ax.scatter([g.exact["t_knn"]], [1.0], color=c, marker=mk, s=70, label="exact", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("kNN time, s (IVF: train + add + search; median over seeds)", color=INK, fontsize=9)
    ax.set_ylabel("recall@15 vs exact", color=INK, fontsize=9)
    ax.set_title(f"{lab}: recall vs kNN time", color=INK, fontsize=10, loc="left")
    _ax_style(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_scaling(groups: list[Group], path: Path) -> bool:
    syn = sorted([g for g in groups if g.key[0] == "synthetic"], key=lambda g: g.key[1])
    if not syn:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm in NND:
        pts = [(g.key[1], med(g.vals(arm, "t_knn"))) for g in syn if g.vals(arm, "t_knn")]
        if pts:
            c, mk = STYLE[arm]
            ax.plot(*zip(*pts), color=c, marker=mk, lw=2, label=f"{arm}")
    for arm in IVF:
        pts = []
        for g in syn:
            r, m = g.ratio(arm)
            t = med([x["t_ivf"] for x in m]) if m else None
            if t is not None and not math.isinf(t):
                pts.append((g.key[1], t))
        if pts:
            c, mk = STYLE[arm]
            ax.plot(*zip(*pts), color=c, marker=mk, lw=2, label=f"{arm} (matched recall; unmatched n omitted)")
    ex = [(g.key[1], g.exact["t_knn"]) for g in syn if g.exact]
    if ex:
        c, mk = STYLE["exact"]
        ax.plot(*zip(*ex), color=c, marker=mk, lw=1.5, ls="--", label="exact (reference)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (synthetic, d = 128)", color=INK, fontsize=9)
    ax.set_ylabel("kNN time, s (median over seeds)", color=INK, fontsize=9)
    ax.set_title("kNN time vs n; IVF at the recall of the faster NN-descent arm", color=INK, fontsize=10, loc="left")
    _ax_style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_e2e(groups: list[Group], labs: dict[tuple[str, int], str], path: Path) -> bool:
    gs = [g for g in groups if any(g.by_arm.get(a) for a in E2E)]
    if not gs:
        return False
    fig, axes = plt.subplots(len(gs), 1, figsize=(7, 1.0 + 1.25 * len(gs)), squeeze=False)
    stage_col = {"t_knn": "#2a78d6", "t_graph": "#eda100", "t_layout": "#1baf7a"}
    for ax, g in zip(axes[:, 0], gs):
        arms = [a for a in E2E if g.by_arm.get(a)]
        left = [0.0] * len(arms)
        for st, c in stage_col.items():
            w = [med(g.vals(a, st)) or 0.0 for a in arms]
            ax.barh(arms, w, left=left, color=c, edgecolor="white", linewidth=2, height=0.7,
                    label=st[2:] if ax is axes[0, 0] else None)
            left = [lft + x for lft, x in zip(left, w)]
        for i, a in enumerate(arms):
            ax.text(left[i], i, f" {left[i]:.1f} s", va="center", fontsize=7, color=INK2)
        ax.set_title(labs[g.key], color=INK, fontsize=9, loc="left")
        ax.invert_yaxis()
        _ax_style(ax)
        ax.grid(axis="y", visible=False)
    axes[-1, 0].set_xlabel("end-to-end time, s (median over seeds), split by stage", color=INK, fontsize=9)
    fig.legend(loc="upper right", fontsize=7, frameon=False, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_quality(groups: list[Group], labs: dict[tuple[str, int], str], path: Path) -> bool:
    """Each arm's metric minus the umap_e2e median, per dataset; grey band = criterion (c)'s +-0.005."""
    gs = [g for g in groups if g.vals("umap_e2e", "np15")]
    if not gs:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 1.2 + 0.7 * len(gs)), squeeze=False)
    seen: set[str] = set()
    for ax, q in zip(axes[0], QUALITY):
        ax.axvspan(-0.005, 0.005, color=GRID, alpha=0.8, lw=0)
        ax.axvline(0, color=INK2, lw=1)
        for i, g in enumerate(gs):
            ref = med(g.vals("umap_e2e", q))
            if ref is None:
                continue
            for j, a in enumerate(E2E):
                v = g.vals(a, q)
                if not v:
                    continue
                y = i + (j - 1.5) * 0.18
                c, mk = STYLE[a]
                ax.plot([min(v) - ref, max(v) - ref], [y, y], color=c, lw=2)
                ax.scatter([med(v) - ref], [y], color=c, marker=mk, s=40, zorder=3, edgecolor="white",
                           linewidth=1.2, label=None if a in seen else a)
                seen.add(a)
        ax.set_yticks(range(len(gs)))
        ax.set_yticklabels([labs[g.key] for g in gs], fontsize=8)
        ax.set_ylim(len(gs) - 0.5, -0.5)
        ax.ticklabel_format(axis="x", useOffset=False, style="plain")
        ax.set_title({"np15": "NP@15", "trustworthiness": "trustworthiness@15",
                      "knn_acc": "kNN accuracy"}[q] + " minus umap_e2e median", color=INK, fontsize=9, loc="left")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        _ax_style(ax)
    handles, names = [], []
    for ax in axes[0]:
        h, n = ax.get_legend_handles_labels()
        for hh, nn in zip(h, n):
            if nn not in names:
                handles.append(hh)
                names.append(nn)
    fig.legend(handles, names, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.text(0.01, 0.01, "dot = median, bar = min-max over seeds; grey band = +-0.005 (criterion c); "
             "trustworthiness on 10k fixed rows; kNN accuracy 5-fold, k=15", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=Path("results/raw"))
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    ok, errors, bad = load(args.raw)
    recs = list(ok.values())
    figdir = args.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    full: dict[tuple[str, int], int] = {}
    for r in recs:
        by_key[key_of(r)].append(r)
        full[key_of(r)] = int(r.get("n_full", r["n"]))
    errs_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for e in errors:
        if e.get("dataset") is not None and e.get("n") is not None:
            errs_by_key[key_of(e)].append(e)
    groups = [Group(k, v, errs_by_key.get(k, [])) for k, v in sorted(by_key.items())]
    labs = {g.key: label(g.key, full) for g in groups}

    df = summary_frame(recs)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "summary.csv", index=False)

    md: list[str] = [
        "# eumap benchmark results",
        "",
        f"Generated by `benchmarks/aggregate.py` on {time.strftime('%Y-%m-%d %H:%M')} from `{args.raw}` "
        f"({len(recs)} ok records, {len(errors)} unresolved error records, {bad} unreadable lines). "
        "Protocol and pre-registered criteria: `benchmarks/PROTOCOL.md`. Times in seconds; "
        "`median [min–max]` over seeds.",
        "",
        "## Runs",
        "",
    ]
    inv = []
    for g in groups:
        hosts = sorted({str(r.get("hostname", "?")) for r in g.all_recs})
        cpus = sorted({r.get("cpu_model", "?") for r in g.all_recs})
        parity = all(r.get("thread_parity", False) for r in g.recs if r["arm"] != "exact")
        arms = ", ".join(f"{a}:{len(g.by_arm[a])}" for a in NND + IVF + E2E if g.by_arm.get(a))
        threads = sorted({r.get("threads_affinity") for r in g.recs})
        inv.append([labs[g.key], f"{g.key[1]}", ", ".join(hosts), "; ".join(cpus), ",".join(map(str, threads)),
                    "yes" if parity else "NO", arms])
    md.append(md_table(["dataset", "n", "host(s)", "CPU", "threads", "thread parity", "records per arm"], inv))
    multi = [g for g in groups if g.excluded]
    if multi:
        md += ["", "**Warning:** records from more than one host. Times are compared only within one node, so every "
               "table and verdict below uses only the host with the most records; the others are excluded: "
               + "; ".join(f"{labs[g.key]}: kept {g.host}, excluded {len(g.excluded)} records from "
                           f"{', '.join(sorted({str(r.get('hostname')) for r in g.excluded}))}" for g in multi) + "."]
    if errors:
        md += ["", "Unresolved errors:", ""]
        md += [f"- `{e.get('run_id')}`: {e.get('error', '')[:200]}" for e in errors]

    verdict_lines: list[str] = []
    evals: dict[tuple[str, int], dict[str, Any]] = {}
    for g in groups:
        lab = labs[g.key]
        md += ["", f"## {lab}", ""]
        # kNN table
        rows = []
        if g.exact:
            ex_lab = "exact (10k sampled queries only)" if g.exact.get("recall_sampled") else "exact"
            rows.append([ex_lab, "1.000", f(g.exact["t_knn"], 2), "–", "–", "–", "–", "–",
                         f(g.exact.get("mem", {}).get("knn", {}).get("peak_rss_gb"), 1)])
        for a in NND:
            if g.by_arm.get(a):
                peaks = [r["mem"]["knn"]["peak_rss_gb"] for r in g.by_arm[a].values()]
                rows.append([a, fr(g.vals(a, "recall"), 4), fr(g.vals(a, "t_knn"), 2), "–", "–", "–", "–", "–",
                             f(max(peaks), 1)])
        for a in IVF:
            if g.by_arm.get(a):
                rs = list(g.by_arm[a].values())
                last = [r["sweep"][-1] for r in rs]
                peaks = [max(v["peak_rss_gb"] for v in r["mem"].values()) for r in rs]
                rows.append([a, f"max {fr([x['recall'] for x in last], 4)} @nprobe {fr([x['nprobe'] for x in last], 0)}",
                             "see sweep", fr(g.vals(a, "t_prepare"), 2), fr(g.vals(a, "t_train"), 2),
                             fr(g.vals(a, "t_add"), 2), fr(g.vals(a, "centroid_mse"), 4),
                             fr(g.vals(a, "imbalance"), 3), f(max(peaks), 1)])
        if rows:
            md += ["### kNN stage", "",
                   md_table(["arm", "recall@15", "t_knn", "t_prepare", "t_train", "t_add", "centroid MSE",
                             "imbalance", "peak RSS GB"], rows)]
        # sweep table
        srows = []
        for a in IVF:
            pts: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for r in g.by_arm.get(a, {}).values():
                for pnt in r["sweep"]:
                    pts[pnt["nprobe"]].append({**pnt, "t_knn": r["t_train"] + r["t_add"] + pnt["t_search"]})
            for npb in sorted(pts):
                v = pts[npb]
                srows.append([a, str(npb), fr([x["recall"] for x in v], 4), fr([x["recall_tie"] for x in v], 4),
                              fr([x["missing_frac"] for x in v], 4), fr([x["t_search"] for x in v], 2),
                              fr([x["t_knn"] for x in v], 2)])
        if srows:
            md += ["", "### IVF nprobe sweep", "",
                   md_table(["arm", "nprobe", "recall", "recall (tie-aware)", "missing", "t_search", "t_train+add+search"],
                            srows)]
        # fixed recall targets (informational, not part of the pre-registered criteria)
        frows = []
        for a in IVF:
            row = [a]
            for target in (0.9, 0.95, 0.99):
                ts = []
                for r in g.by_arm.get(a, {}).values():
                    pt = next((q for q in sorted(r["sweep"], key=lambda q: q["nprobe"]) if q["recall"] >= target), None)
                    ts.append(r["t_train"] + r["t_add"] + pt["t_search"] if pt else math.inf)
                row.append(fr(ts, 2) if ts else "–")
            if len(row) > 1 and any(c != "–" for c in row[1:]):
                frows.append(row)
        for a in NND:
            if g.by_arm.get(a):
                rm = med(g.vals(a, "recall"))
                frows.append([a + f" (single point, recall {fr(g.vals(a, 'recall'), 4)})"]
                             + [fr(g.vals(a, "t_knn"), 2) if rm >= t else "not reached" for t in (0.9, 0.95, 0.99)])
        if frows:
            md += ["", "### kNN time to reach a fixed recall (informational)", "",
                   "IVF: train + add + search at the smallest swept nprobe reaching the recall (inf: not reached "
                   "within the sweep). NN-descent has one operating point, shown for reference.", "",
                   md_table(["arm", "recall >= 0.90", "recall >= 0.95", "recall >= 0.99"], frows)]
        # matched recall
        mrows = []
        for nnd in NND:
            for a in IVF:
                m = g.matched(a, nnd)
                if not m:
                    continue
                lbs = [x["lb_ratio"] for x in m if x["lb_ratio"] is not None]
                mrows.append([nnd, a, fr([x["r_nnd"] for x in m], 4),
                              ("capped " if any(x["capped"] for x in m) else "") + fr([x["target"] for x in m], 4),
                              ",".join(str(x["nprobe"]) if x["nprobe"] else f"unmatched ({x['unmatched']})" for x in m),
                              fr([x["t_ivf"] for x in m], 2), fr([x["t_nnd"] for x in m], 2),
                              fr([x["ratio"] for x in m], 3), fr([x["ratio_interp"] for x in m], 3),
                              (">= " + fr(lbs, 3)) if lbs else "–"])
        if mrows:
            fast = g.faster_nnd()
            md += ["", f"### Matched recall (faster NN-descent arm: `{fast}`)", "",
                   md_table(["NN-descent arm", "IVF arm", "NN-descent recall", "target", "nprobe per seed",
                             "IVF kNN time", "NN-descent time", "ratio IVF/NND",
                             "ratio, interpolated (informational)", "unmatched: ratio lower bound"], mrows),
                   "", "The pre-registered ratio uses the smallest swept power-of-two nprobe that reaches the target. "
                   "*Interpolated* charges IVF the search time interpolated linearly in recall between the two "
                   "swept nprobes that bracket the target, to show how much of the ratio is grid overshoot. "
                   "*Unmatched (cap)* means the sweep stopped at nprobe = 256 (the spec's limit) below the target; "
                   "*(nlist)* means every list was probed. For those seeds the ratio is inf, and the IVF cost at "
                   "the last swept nprobe gives a lower bound on what a longer sweep could have reached."]
        # e2e
        erows = []
        for a in E2E:
            if g.by_arm.get(a):
                extra = ""
                if a in ("eumap_e2e", "eumap_lloyd_e2e"):
                    extra = "nprobe " + ",".join(str(r["knn_info"]["nprobe"]) for r in g.by_arm[a].values())
                erows.append([a, fr(g.vals(a, "t_e2e"), 1), fr(g.vals(a, "t_knn"), 1), fr(g.vals(a, "t_graph"), 1),
                              fr(g.vals(a, "t_layout"), 1), fr(g.vals(a, "graph_recall"), 4),
                              fr(g.vals(a, "np15"), 4), fr(g.vals(a, "trustworthiness"), 4),
                              fr(g.vals(a, "knn_acc"), 4), extra])
        if erows:
            md += ["", "### End to end and embedding quality", "",
                   md_table(["arm", "t_e2e", "t_knn", "t_graph", "t_layout", "graph recall", "NP@15",
                             "trustworthiness", "kNN acc", "notes"], erows)]
        # figure
        fig_name = f"recall_vs_time_{lab.replace(' ', '_').replace('=', '')}.png"
        if any(g.by_arm.get(a) for a in NND + IVF):
            fig_recall_time(g, lab, figdir / fig_name)
            md += ["", f"![recall vs kNN time](figures/{fig_name})"]
        # verdict lines
        ev = evaluate(g)
        evals[g.key] = ev
        in_eval = any(g.key[0] == ds and (n is None or g.key[1] == n) for ds, n in EVAL_SET)
        tag = "" if in_eval else " [informational: not in the evaluation set]"
        lines = [f"{lab}{tag}:"]
        if g.excluded:
            lines.append(f"  note: only host {g.host} is used; {len(g.excluded)} records from "
                         f"{', '.join(sorted({str(r.get('hostname')) for r in g.excluded}))} are excluded")
        for kk in ("a", "b", "c"):
            lines.append(f"  {yn(ev[kk][0]):>3}  {ev[kk][1]}")
        lines.append(f"  {yn(ev['worth']):>3}  (a)+(b)+(c) all hold (eUMAP worth using on this dataset)")
        for kk in ("a_lloyd", "b_lloyd", "c_lloyd"):
            lines.append(f"  {yn(ev[kk][0]):>3}  [Lloyd variant] {ev[kk][1]}")
        lines.append(f"  {yn(ev['emeans'][0]):>3}  {ev['emeans'][1]}")
        lines.append(f"  {yn(ev['ivf_helps_emeans_not']):>3}  the Faiss IVF graph helps, e-means does not "
                     f"(e-means end-to-end gain {f(ev['gain'])}, needs < 0.05 with (a)-(c) met by an IVF pair)")
        beats = ", ".join(f"{a} {f(r)}" + (f" (>= {f(ev['beats_opt'][a])} at the end of the sweep)"
                                           if math.isinf(r) and ev["beats_opt"].get(a) is not None
                                           and not math.isinf(ev["beats_opt"][a]) else "")
                          for a, r in ev["beats"].items()) or "no data"
        lines.append(f"  {yn(any(r < 1 for r in ev['beats'].values()) if ev['beats'] else None):>3}  "
                     f"some IVF arm beats the faster NN-descent arm at matched recall (median ratios: {beats})")
        verdict_lines += lines
        md += ["", "### Verdict lines", "", "```", *lines, "```"]

    # overall verdicts
    overall: list[str] = ["OVERALL (pre-registered reading, benchmarks/PROTOCOL.md):"]
    eval_keys = [g.key for g in groups if any(g.key[0] == ds and (n is None or g.key[1] == n) for ds, n in EVAL_SET)]
    has_gist = any(k[0] == "gist1m" for k in eval_keys)
    need, planned = (4, 5) if has_gist else (3, 4)
    evaluated = [k for k in eval_keys if evals[k]["worth"] is not None]
    passed = [k for k in evaluated if evals[k]["worth"]]
    worth = len(passed) >= need
    status = "" if len(evaluated) >= planned else f" [INCOMPLETE: {len(evaluated)} of {planned} evaluated]"
    overall.append(f"  {('YES' if worth else 'no') if evaluated else 'n/a':>3}  eUMAP is worth using: (a)-(c) hold on {len(passed)} of "
                   f"{len(evaluated)} evaluated ({', '.join(labs[k] for k in passed) or 'none'}); needs {need} of "
                   f"{planned}{status}")
    # Both e-means lines use the same cross-dataset rule as "worth using" (PROTOCOL.md change log).
    def incomplete(n_eval: int) -> str:
        return "" if n_eval >= planned else f" [INCOMPLETE: {n_eval} of {planned} evaluated]"

    ec_eval = [k for k in eval_keys if evals[k]["emeans"][0] is not None]
    ec_yes = [k for k in ec_eval if evals[k]["emeans"][0]]
    overall.append(f"  {('YES' if len(ec_yes) >= need else 'no') if ec_eval else 'n/a':>3}  e-means "
                   f"contributes: holds on {len(ec_yes)} of {len(ec_eval)} evaluated "
                   f"({', '.join(labs[k] for k in ec_yes) or 'none'}); needs {need} of {planned}"
                   f"{incomplete(len(ec_eval))}")
    ih = [k for k in eval_keys if evals[k]["ivf_helps_emeans_not"]]
    ih_eval = [k for k in eval_keys if evals[k]["ivf_helps_emeans_not"] is not None]
    overall.append(f"  {('YES' if len(ih) >= need else 'no') if ih_eval else 'n/a':>3}  'the Faiss IVF graph helps, "
                   f"e-means does not': holds on {len(ih)} of {len(ih_eval)} evaluated "
                   f"({', '.join(labs[k] for k in ih) or 'none'}); needs {need} of {planned}"
                   f"{incomplete(len(ih_eval))}")
    small = [g.key for g in groups if g.key[1] <= 1_000_000 and evals[g.key]["beats"]]
    small_beat = [k for k in small if any(r < 1 for r in evals[k]["beats"].values())]
    q_fail = [k for k in eval_keys if evals[k]["c"][0] is False]
    not_worth = (bool(small) and not small_beat) or bool(q_fail)
    q_eval = [k for k in eval_keys if evals[k]["c"][0] is not None]
    partial = (f" [PARTIAL: {len(small)} datasets with n <= 1e6 and {len(q_eval)} of {planned} evaluation datasets "
               "have the needed records]")
    overall.append(f"  {('YES' if not_worth else 'no') if small or q_eval else 'n/a':>3}  Not worth using: IVF beats "
                   f"NN-descent at matched recall on {len(small_beat)} of {len(small)} datasets with n <= 1e6 "
                   f"({', '.join(labs[k] for k in small_beat) or 'none'}); quality (c) fails on "
                   f"{', '.join(labs[k] for k in q_fail) or 'none'} of {len(q_eval)} evaluated{partial}")
    if worth and not_worth:
        overall.append("  CONFLICT: both 'worth using' and 'not worth using' hold; both are reported, not resolved.")
    small_opt = [k for k in small if k not in small_beat
                 and any(r is not None and r < 1 for r in evals[k]["beats_opt"].values())]
    if small_opt:
        overall.append(f"  note: on {', '.join(labs[k] for k in small_opt)} no IVF arm reached NN-descent's recall "
                       "within nprobe <= 256 although its cost there was below NN-descent's; there the 'beats' "
                       "count is decided by the nprobe cap (see the matched-recall tables)")
    syn = sorted([g for g in groups if g.key[0] == "synthetic"], key=lambda g: g.key[1])
    def crossover(ns: list[tuple[int, float]]) -> int | None:
        for i, (n, _) in enumerate(ns):
            if all(r < 1 for _, r in ns[i:]):
                return n
        return None

    def best(d: dict[str, float | None]) -> float | None:
        xs = [x for x in d.values() if x is not None]
        return min(xs) if xs else None

    for arm_desc, pick in (("ivf_emeans", lambda d: d.get("ivf_emeans")), ("best IVF arm", best)):
        ns = [(g.key[1], pick(evals[g.key]["beats"])) for g in syn]
        ns = [(n, r) for n, r in ns if r is not None]
        opt = dict((g.key[1], pick(evals[g.key]["beats_opt"])) for g in syn)
        cross = crossover(ns)
        detail = ", ".join(f"n={fmt_n(n)}: {f(r)}" + (f" (>= {f(opt[n])} at the end of the sweep)"
                                                      if math.isinf(r) and opt.get(n) is not None
                                                      and not math.isinf(opt[n]) else "")
                           for n, r in ns) or "no data"
        overall.append(f"  synthetic crossover ({arm_desc} vs faster NN-descent): "
                       + (f"IVF overtakes from n = {fmt_n(cross)}" if cross else
                          f"none up to {fmt_n(max(n for n, _ in ns))}" if ns else "no data")
                       + f"  (median ratios: {detail})")
        cross_opt = crossover([(n, opt[n]) for n, _ in ns if opt.get(n) is not None])
        if ns and cross_opt != cross:
            overall.append(f"    note: with end-of-sweep costs as lower bounds for the unmatched seeds the crossover "
                           f"could be as early as n = {fmt_n(cross_opt) if cross_opt else 'none'}; the line above is "
                           "decided by the nprobe cap of 256 there")
    verdict_lines += overall
    md[md.index("## Runs"):md.index("## Runs")] = ["## Verdicts", "", "```", *overall, "```", ""]

    figs = []
    if fig_scaling(groups, figdir / "scaling_knn_time.png"):
        figs.append("![scaling](figures/scaling_knn_time.png)")
    if fig_e2e(groups, labs, figdir / "e2e_stages.png"):
        figs.append("![end-to-end stages](figures/e2e_stages.png)")
    if fig_quality(groups, labs, figdir / "quality.png"):
        figs.append("![quality](figures/quality.png)")
    if figs:
        md += ["", "## Summary figures", "", *figs]
    (args.out / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(verdict_lines))
    print(f"\nwrote {args.out / 'summary.csv'} ({len(df)} rows), {args.out / 'RESULTS.md'}, figures in {figdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
