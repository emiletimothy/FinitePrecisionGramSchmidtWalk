"""Experiment grid, parallel execution and the raw-result cache.

Each task writes one file under <outdir>/raw/. Existing files are skipped, so an
interrupted grid resumes where it stopped, and the report is rebuilt from raw/.
"""
import json
import os
import time
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from . import matrices, metrics, walks
from .diagnostics import FIELDS, StepLogger

DTYPES = ("float64", "float32", "float16")
FAMILIES = ("gaussian", "conditioned", "collinear", "harshaw")
KAPPAS = (1e0, 1e2, 1e4, 1e6, 1e8)
ETAS = (1e-1, 1e-3, 1e-5, 1e-7)
HARSHAW_D = (5, 50)
PHIS = (0.5, 0.1, 0.01)
IMPLS = {
    "gaussian": ("lstsq", "gs"),
    "conditioned": ("lstsq", "gs"),
    "collinear": ("lstsq", "gs"),
    "harshaw": ("lstsq", "chol", "chol_rk10", "gs"),
}
CHUNK = {50: 100, 200: 20, 500: 4}   # lstsq runs per task; cheaper impls get 5x


def configs(ns, families):
    out = []
    for n in ns:
        for fam in families:
            if fam == "gaussian":
                out.append((fam, n, {}))
            elif fam == "conditioned":
                out += [(fam, n, {"kappa": k}) for k in KAPPAS]
            elif fam == "collinear":
                out += [(fam, n, {"eta": e}) for e in ETAS]
            elif fam == "harshaw":
                out += [(fam, n, {"d": d, "phi": phi, "corr": c})
                        for d in HARSHAW_D for phi in PHIS for c in (False, True)]
    return out


def fmt_param(v):
    return str(int(v)) if isinstance(v, bool) else f"{v:g}"


def param_str(params):
    return ",".join(f"{k}={fmt_param(v)}" for k, v in sorted(params.items()))


def config_key(family, n, params):
    return "_".join([family, f"n{n}"] + [f"{k}={fmt_param(v)}" for k, v in sorted(params.items())])


@lru_cache(maxsize=8)
def instance(family, n, params_json, seed):
    inst = matrices.make_instance(family, n, json.loads(params_json), seed)
    V, names = metrics.test_directions(inst.B)
    return inst, V, names


def call_impl(impl, inst, U, dtype, **kw):
    if impl == "lstsq":
        return walks.walk_lstsq(inst.B, U, dtype, **kw)
    if impl == "gs":
        return walks.walk_gs_compress(inst.B, U, dtype, **kw)
    if impl.startswith("chol"):
        k = int(impl[len("chol_rk"):]) if impl.startswith("chol_rk") else None
        return walks.walk_harshaw_cholesky(inst.X, inst.phi, U, dtype, refactor_every=k, **kw)
    raise ValueError(impl)


def _save(path, meta, arrays):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def run_task(task):
    meta = task["meta"]
    fam, n, seed, impl, dtype = meta["family"], meta["n"], meta["seed"], meta["impl"], meta["dtype"]
    inst, V, _ = instance(fam, n, json.dumps(meta["params"], sort_keys=True), seed)
    runs = np.arange(task["lo"], task["hi"])
    t0 = time.perf_counter()

    if meta["kind"] == "diag":
        rows = []
        B_dt = inst.B.astype(dtype)
        for run in runs:
            logger = StepLogger(B_dt, run)
            call_impl(impl, inst, walks.uniforms(seed, run, n), dtype, logger=logger)
            rows.append(logger.array())
        _save(Path(task["path"]), meta, {"diag": np.concatenate(rows), "fields": np.array(FIELDS)})
        return task["path"], time.perf_counter() - t0

    R, L = len(runs), 2 * n + 4
    out = {
        "runs": runs,
        "Y": np.full((R, V.shape[1]), np.nan),
        "failed": np.empty(R, dtype="U32"),
        "steps": np.zeros(R, dtype=np.int16),
        "pivots": np.full((R, L), -1, dtype=np.int16),
        "hit_step": np.full((R, n), -1, dtype=np.int16),
        "signs": np.zeros((R, n), dtype=np.int8),
        "forced_max": np.zeros(R), "tol_snaps": np.zeros(R, dtype=np.int32),
        "tol_max": np.zeros(R), "clamps": np.zeros(R, dtype=np.int32),
        "clamp_max": np.zeros(R), "zero_steps": np.zeros(R, dtype=np.int32),
        "seconds": np.zeros(R),
    }
    for j, run in enumerate(runs):
        res = call_impl(impl, inst, walks.uniforms(seed, run, n), dtype)
        out["failed"][j] = res.failed or ""
        out["steps"][j] = res.steps
        out["pivots"][j, :len(res.pivots)] = res.pivots
        out["hit_step"][j] = res.hit_step
        for f in ("forced_max", "tol_snaps", "tol_max", "clamps", "clamp_max", "zero_steps", "seconds"):
            out[f][j] = getattr(res, f)
        if res.failed is None:
            z = res.z.astype(np.float64)
            out["signs"][j] = z
            out["Y"][j] = (inst.B @ z) @ V     # z0 = 0, so D = B z
    _save(Path(task["path"]), meta, out)
    return task["path"], time.perf_counter() - t0


def build_tasks(ns, families, dtypes, impls, R_for, diag_for, seed, outdir):
    raw = Path(outdir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tasks = []
    for fam, n, params in configs(ns, families):
        key = config_key(fam, n, params)
        for impl in IMPLS[fam]:
            if impls and impl not in impls:
                continue
            for dtype in dtypes:
                meta = dict(family=fam, n=n, params=params, key=key, impl=impl, dtype=dtype, seed=seed)
                R = R_for(impl, dtype, n)
                chunk = CHUNK.get(n, 10) * (1 if impl == "lstsq" else 5)
                for lo in range(0, R, chunk):
                    hi = min(R, lo + chunk)
                    path = raw / f"{key}__{impl}__{dtype}__{lo:05d}-{hi:05d}.npz"
                    tasks.append(dict(meta=dict(meta, kind="runs"), lo=lo, hi=hi, path=str(path)))
                nd = diag_for(n)
                if nd:
                    path = raw / f"diag__{key}__{impl}__{dtype}__{nd:03d}.npz"
                    tasks.append(dict(meta=dict(meta, kind="diag"), lo=0, hi=nd, path=str(path)))
    return tasks


# Rough idle seconds per run: (seconds at n=200, exponent in n, float16 multiplier), fitted
# to the timing projection on the development laptop. Only the ratios matter: they order
# tasks (slowest first) and fill in --status for cells with no finished file yet.
_PER_RUN = {"lstsq": (1.3, 3.2, 5.0), "gs": (0.03, 3.0, 8.0),
            "chol": (0.17, 1.1, 1.3), "chol_rk10": (0.2, 1.1, 1.6)}


def _cost(task):
    m = task["meta"]
    a, p, f16 = _PER_RUN[m["impl"]]
    s = a * (m["n"] / 200) ** p * (f16 if m["dtype"] == "float16" else 1.0)
    if m["kind"] == "diag":
        s *= DIAG_FACTOR
    return s * (task["hi"] - task["lo"])


def execute(tasks, workers):
    todo = sorted((t for t in tasks if not Path(t["path"]).exists()), key=_cost, reverse=True)
    print(f"{len(tasks) - len(todo)} of {len(tasks)} tasks cached; running {len(todo)} on {workers} workers",
          flush=True)
    if not todo:
        return
    t0 = time.time()
    last = 0.0
    with get_context("forkserver").Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(run_task, todo), 1):
            el = time.time() - t0
            if el - last > 30 or i == len(todo):
                last = el
                print(f"  {i}/{len(todo)} tasks, {el / 60:.1f} min elapsed", flush=True)


# ---------------------------------------------------------------- progress of a running grid

DIAG_FACTOR = 3.0   # a diagnostic run adds a float64 reference solve and an SVD per step


def status(tasks, workers):
    """Estimate time remaining from the per-run seconds recorded in finished files.

    Those seconds were measured while all workers were busy, so remaining
    CPU-seconds / workers approximates wall-clock time. Cells with no finished
    file yet fall back to `_cost`, scaled to the measured tasks.
    """
    measured, ratios, done_cpu = {}, [], 0.0
    for t in tasks:
        p = Path(t["path"])
        if t["meta"]["kind"] != "runs" or not p.exists():
            continue
        with np.load(p) as d:
            s = d["seconds"]
        m = t["meta"]
        measured.setdefault((m["impl"], m["dtype"], m["n"]), []).extend(s)
        done_cpu += s.sum()
        ratios.append(s.sum() / _cost(t))
    if not ratios:
        print("No finished run files yet; check again once some tasks have completed.")
        return
    fallback = float(np.median(ratios))
    left_meas = left_guess = 0.0
    n_left = 0
    for t in tasks:
        if Path(t["path"]).exists():
            continue
        n_left += 1
        m = t["meta"]
        cell = (m["impl"], m["dtype"], m["n"])
        factor = DIAG_FACTOR if m["kind"] == "diag" else 1.0
        if cell in measured:
            left_meas += factor * (t["hi"] - t["lo"]) * float(np.mean(measured[cell]))
        else:
            left_guess += _cost(t) * fallback
    left = left_meas + left_guess
    print(f"{len(tasks) - n_left}/{len(tasks)} tasks finished, {done_cpu / 3600:.1f} CPU-h of walks done")
    print(f"remaining ~{left / 3600:.1f} CPU-h: {left_meas / 3600:.1f} from measured per-run times, "
          f"{left_guess / 3600:.1f} from the rough cost model")
    print(f"estimated time left on {workers} workers: ~{left / workers / 3600:.1f} h "
          f"(tasks in progress are counted as not started)")


# ---------------------------------------------------------------- timing projection

def _representative(impl, n, seed):
    if impl.startswith("chol"):
        return instance("harshaw", n, json.dumps({"corr": False, "d": 50, "phi": 0.1}, sort_keys=True), seed)[0]
    return instance("gaussian", n, "{}", seed)[0]


def seconds_per_run(impl, dtype, n, seed=0):
    """Time 1 and 11 steps, then extrapolate: per-step cost ~ k^2 for lstsq/gs, ~ const for chol."""
    inst = _representative(impl, n, seed)
    U = walks.uniforms(seed, 0, n)
    t1 = call_impl(impl, inst, U, dtype, max_steps=1).seconds
    t11 = call_impl(impl, inst, U, dtype, max_steps=11).seconds
    per_step = max(t11 - t1, 0.0) / 10
    return t1 + per_step * (n / 3 if impl in ("lstsq", "gs") else n)


def project(ns, families, dtypes, impls, R_for, workers, seed=0):
    tasks = [t for t in build_tasks(ns, families, dtypes, impls, R_for, lambda n: 0, seed, "/tmp/unused")
             if t["meta"]["kind"] == "runs"]
    runs = {}
    for t in tasks:
        m = t["meta"]
        k = (m["impl"], m["dtype"], m["n"])
        runs[k] = runs.get(k, 0) + t["hi"] - t["lo"]
    print("\nTiming projection (seconds per run from 1- and 11-step partial walks):")
    print(f"{'impl':>10} {'dtype':>8} {'n':>5} {'s/run':>9} {'runs':>8} {'CPU-h':>8}")
    total = 0.0
    for (impl, dtype, n), R in sorted(runs.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        s = seconds_per_run(impl, dtype, n, seed)
        total += s * R
        print(f"{impl:>10} {dtype:>8} {n:>5} {s:>9.3f} {R:>8} {s * R / 3600:>8.2f}")
    print(f"Projected wallclock on {workers} workers: {total / 3600 / workers:.1f} h "
          f"(excludes per-step diagnostics)")
    return total / workers
