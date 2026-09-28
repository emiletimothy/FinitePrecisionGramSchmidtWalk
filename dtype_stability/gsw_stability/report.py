"""Results table (CSV + markdown) built from <outdir>/raw/."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import metrics
from .runner import DTYPES, FAMILIES, param_str

IMPL_ORDER = ("lstsq", "chol", "chol_rk10", "gs")
COLUMNS = [
    "family", "param", "n", "impl", "dtype", "R", "fail_frac",
    "psi2_orlicz_max", "psi2_orlicz_med", "psi2_moment_max", "psi2_bottom_sv", "psi2_ratio",
    "max_bias", "bias_se", "diverged_frac", "med_first_div",
    "max_err_Bu", "max_err_u", "max_err_p", "max_drift",
    "tol_snaps_per_run", "clamps_per_run", "forced_max", "max_steps", "sec_per_run", "fail_reasons",
]


def index(outdir):
    """{(key, impl, dtype): [run files]} and {(key, impl, dtype): diag file}."""
    runs, diags = defaultdict(list), {}
    for f in sorted((Path(outdir) / "raw").glob("*.npz")):
        parts = f.stem.split("__")
        if parts[0] == "diag":
            k = tuple(parts[1:4])
            if k not in diags or f.stem > diags[k].stem:
                diags[k] = f
        else:
            runs[tuple(parts[:3])].append(f)
    return runs, diags


def load_runs(files):
    meta, chunks = None, []
    for f in files:
        with np.load(f) as d:
            meta = json.loads(str(d["meta"]))
            chunks.append({k: d[k] for k in d.files if k != "meta"})
    out = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    _, first = np.unique(out["runs"], return_index=True)
    return meta, {k: v[first] for k, v in out.items()}


def load_diag(path):
    with np.load(path) as d:
        return {name: d["diag"][:, j] for j, name in enumerate(d["fields"])}


def _nanmax(a):
    a = a[np.isfinite(a)]
    return a.max() if a.size else np.nan


def _row(meta, g, base, diag):
    ok = g["failed"] == ""
    Y = g["Y"][ok]
    row = dict(family=meta["family"], param=param_str(meta["params"]), n=meta["n"],
               impl=meta["impl"], dtype=meta["dtype"], R=len(ok), fail_frac=1 - ok.mean(),
               fail_reasons=";".join(f"{k}:{v}" for k, v in Counter(g["failed"][~ok]).most_common()))
    if len(Y) >= 2:
        s = metrics.summarize(Y)
        j = np.argmax(np.abs(s["mean"]))
        row.update(psi2_orlicz_max=s["orlicz"].max(), psi2_orlicz_med=np.median(s["orlicz"]),
                   psi2_moment_max=s["moment"].max(), psi2_bottom_sv=s["orlicz"][-1],
                   max_bias=abs(s["mean"][j]), bias_se=s["se"][j])
    if base is not None:
        common, ia, ib = np.intersect1d(g["runs"], base["runs"], return_indices=True)
        both = (g["failed"][ia] == "") & (base["failed"][ib] == "")
        ia, ib = ia[both], ib[both]
        if len(ia) >= 2:
            psi = lambda Y: max(metrics.psi2_orlicz(Y[:, v]) for v in range(Y.shape[1]))
            row["psi2_ratio"] = psi(g["Y"][ia]) / psi(base["Y"][ib])
        fd = np.array([metrics.first_divergence(g["hit_step"][a], g["pivots"][a],
                                                base["hit_step"][b], base["pivots"][b])
                       for a, b in zip(ia, ib)])
        if len(fd):
            row["diverged_frac"] = np.mean(fd >= 0)
            row["med_first_div"] = np.median(fd[fd >= 0]) if np.any(fd >= 0) else np.nan
    if diag is not None:
        row.update(max_err_Bu=_nanmax(diag["err_Bu"]), max_err_u=_nanmax(diag["err_u"]),
                   max_err_p=_nanmax(diag["err_p"]),
                   max_drift=_nanmax(np.concatenate([diag["drift"], diag["inv_drift"]])))
    row.update(tol_snaps_per_run=g["tol_snaps"].mean(), clamps_per_run=g["clamps"].mean(),
               forced_max=g["forced_max"].max(), max_steps=int(g["steps"].max()),
               sec_per_run=g["seconds"].mean())
    return row


def _sort_key(r):
    return (r["n"], FAMILIES.index(r["family"]), r["param"],
            IMPL_ORDER.index(r["impl"]), DTYPES.index(r["dtype"]))


def build_table(outdir):
    runs, diags = index(outdir)
    rows = []
    for key in sorted({k[0] for k in runs}):
        base = None
        if (key, "lstsq", "float64") in runs:
            _, base = load_runs(runs[(key, "lstsq", "float64")])
        for (k, impl, dtype), files in runs.items():
            if k != key:
                continue
            meta, g = load_runs(files)
            d = diags.get((key, impl, dtype))
            rows.append(_row(meta, g, base, load_diag(d) if d else None))
    return sorted(rows, key=_sort_key)


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.3g}"
    return str(v)


def to_markdown(rows, columns=COLUMNS):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_fmt(r.get(c, np.nan)) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def write(outdir):
    rows = build_table(outdir)
    outdir = Path(outdir)
    with open(outdir / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c, np.nan)) for c in COLUMNS})
    md = to_markdown(rows)
    (outdir / "table.md").write_text(md + "\n")
    return rows, md
