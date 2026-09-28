"""Figures built from <outdir>/raw (never from in-memory state).

Colour encodes precision (three fixed slots) or, in the step-error scatter,
implementation; panels split everything else, so no axis carries more than
three coloured series.
"""
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import report
from .runner import DTYPES, config_key

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DTYPE_COLOR = {"float64": "#2a78d6", "float32": "#eb6834", "float16": "#1baf7a"}
IMPL_COLOR = {"lstsq": "#2a78d6", "chol": "#eb6834", "gs": "#1baf7a"}
EPS = {d: float(np.finfo(d).eps) for d in DTYPES}
LOW = ("float32", "float16")


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.5, "lines.markersize": 5,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
        "figure.constrained_layout.use": True,
    })


def _params(r):
    return {k: float(v) for k, v in (kv.split("=") for kv in r["param"].split(",") if kv)}


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _get(r, k):
    v = r.get(k, np.nan)
    return np.nan if v is None else float(v)


# ---------------------------------------------------------------- psi_2 sweeps

def sweep_plots(rows, figdir):
    for family, pname, xlabel in [("conditioned", "kappa", "κ(B)"), ("collinear", "eta", "η (pair separation)")]:
        for n in sorted({r["n"] for r in rows if r["family"] == family}):
            rs = [r for r in rows if r["family"] == family and r["n"] == n]
            impls = [i for i in report.IMPL_ORDER if any(r["impl"] == i for r in rs)]
            fig, axes = plt.subplots(3, len(impls), figsize=(3.8 * len(impls), 7.5), squeeze=False, sharex=True)
            for c, impl in enumerate(impls):
                for dtype in DTYPES:
                    sel = sorted((r for r in rs if r["impl"] == impl and r["dtype"] == dtype),
                                 key=lambda r: _params(r)[pname])
                    if not sel:
                        continue
                    x = [_params(r)[pname] for r in sel]
                    kw = dict(color=DTYPE_COLOR[dtype], marker="o", label=dtype)
                    axes[0, c].plot(x, [_get(r, "psi2_orlicz_max") for r in sel], **kw)
                    axes[1, c].errorbar(x, [_get(r, "max_bias") for r in sel],
                                        yerr=[_get(r, "bias_se") for r in sel], capsize=2, **kw)
                    axes[2, c].plot(x, [_get(r, "fail_frac") for r in sel], **kw)
                axes[0, c].set_title(f"{impl}, n={n}")
                axes[2, c].set_xlabel(xlabel)
                for a in axes[:, c]:
                    a.set_xscale("log")
                axes[0, c].set_yscale("log")
                axes[1, c].set_yscale("log")
                axes[2, c].set_ylim(-0.03, 1.03)
            axes[0, 0].set_ylabel("ψ₂ (Orlicz), max over directions")
            axes[1, 0].set_ylabel("max |mean Y_v|  (bar: ±1 SE)")
            axes[2, 0].set_ylabel("fraction of runs failed")
            axes[0, 0].legend(title="dtype")
            _save(fig, figdir / f"psi2_vs_{pname}_n{n}.png")


# ---------------------------------------------------------------- tail CCDFs

REPRESENTATIVE = [
    ("gaussian", {}),
    ("conditioned", {"kappa": 1e4}),
    ("conditioned", {"kappa": 1e8}),
    ("collinear", {"eta": 1e-5}),
    ("harshaw", {"d": 50, "phi": 0.01, "corr": True}),
]


CCDF_NS = (50, 200)   # n=500 has too few lstsq runs for a tail


def ccdf_plots(outdir, figdir):
    """|Y_v| scaled by the float64 lstsq standard deviation of Y_v, so exp(-t^2/2)
    is the tail of a Gaussian with the baseline's variance in every panel."""
    runs, _ = report.index(outdir)
    for family, params in REPRESENTATIVE:
        for n in CCDF_NS:
            key = config_key(family, n, params)
            if (key, "lstsq", "float64") not in runs:
                continue
            _, gb = report.load_runs(runs[(key, "lstsq", "float64")])
            sd = gb["Y"][gb["failed"] == ""].std(axis=0)
            impls = [i for i in report.IMPL_ORDER if any((key, i, d) in runs for d in DTYPES)]
            fig, axes = plt.subplots(2, len(impls), figsize=(3.8 * len(impls), 6.4), squeeze=False)
            ymin = 1.0
            for c, impl in enumerate(impls):
                for dtype in DTYPES:
                    if (key, impl, dtype) not in runs:
                        continue
                    _, g = report.load_runs(runs[(key, impl, dtype)])
                    Y = g["Y"][g["failed"] == ""]
                    if not len(Y):
                        continue
                    ymin = min(ymin, 0.5 / len(Y))
                    for r, v in enumerate((0, -1)):
                        t = np.sort(np.abs(Y[:, v]) / sd[v])[::-1]
                        axes[r, c].step(t, np.arange(1, len(t) + 1) / len(t), where="post",
                                        color=DTYPE_COLOR[dtype], label=f"{dtype} (R={len(Y)})")
                axes[0, c].set_title(f"{impl}: random v")
                axes[1, c].set_title(f"{impl}: bottom singular vector")
            for a in axes.flat:
                tt = np.linspace(0, max(a.get_xlim()[1], 3.5), 200)
                a.plot(tt, np.exp(-tt**2 / 2), color=MUTED, lw=1.2, label="exp(−t²/2)")
                a.set_yscale("log")
                a.set_ylim(ymin, 1.05)
                a.set_xlabel("|Y_v| / sd of float64 lstsq")
            for r in range(2):
                axes[r, 0].set_ylabel("P(|Y_v| / sd ≥ t)")
            for a in axes[0]:
                a.legend(fontsize=7)
            fig.suptitle(f"Tail of ⟨B z, v⟩: {key}", color=INK, fontsize=10)
            _save(fig, figdir / f"ccdf_{key}.png")


# ---------------------------------------------------------------- step diagnostics

def _load_diags(outdir):
    _, diags = report.index(outdir)
    return {k: report.load_diag(f) for k, f in diags.items()}


KAPPA_MAX = 1e17   # past ~1/eps(float64) the reference SVD cannot resolve kappa


def step_error_vs_kappa(diags, figdir):
    cols = [("err_Bu", "‖B(û − u⁽⁶⁴⁾)‖₂  (discrepancy space)"),
            ("err_u", "‖û − u⁽⁶⁴⁾‖∞  (coefficients)"),
            ("rel_u", "‖û − u⁽⁶⁴⁾‖∞ / ‖u⁽⁶⁴⁾‖∞")]
    fig, axes = plt.subplots(len(LOW), 3, figsize=(12, 7.5), squeeze=False)
    for r, dtype in enumerate(LOW):
        kmax, hidden, ys = 1.0, 0, [[] for _ in cols]
        for impl, color in IMPL_COLOR.items():
            ds = [d for (key, i, dt), d in diags.items() if i == impl and dt == dtype]
            if not ds:
                continue
            kap = np.concatenate([d["kappa"] for d in ds])
            vals = {"err_Bu": np.concatenate([d["err_Bu"] for d in ds]),
                    "err_u": np.concatenate([d["err_u"] for d in ds])}
            vals["rel_u"] = vals["err_u"] / np.concatenate([d["norm_u"] for d in ds])
            hidden += int(np.sum(np.isfinite(kap) & (kap > KAPPA_MAX)))
            for c, (f, _) in enumerate(cols):
                ok = np.isfinite(kap) & (kap <= KAPPA_MAX) & np.isfinite(vals[f]) & (vals[f] > 0)
                axes[r, c].scatter(kap[ok], vals[f][ok], s=4, alpha=0.3, color=color, lw=0, label=impl)
                if ok.any():
                    kmax = max(kmax, kap[ok].max())
                    ys[c].append(vals[f][ok])
        kk = np.logspace(0, np.log10(kmax), 50)
        for c, (f, title) in enumerate(cols):
            a = axes[r, c]
            a.set_xscale("log")
            a.set_yscale("log")
            a.set_xlim(0.8, kmax * 1.5)
            if ys[c]:
                lo, hi = np.percentile(np.concatenate(ys[c]), [0.05, 99.95])
                a.set_ylim(lo / 3, hi * 3)
            ytop = a.get_ylim()[1]
            for power, lab in [(1, "ε·κ"), (2, "ε·κ²")]:
                a.plot(kk, EPS[dtype] * kk**power, color=MUTED, lw=1, scalex=False, scaley=False)
                xl = min(kk[-1], (ytop / EPS[dtype]) ** (1 / power))
                a.annotate(lab, (xl, EPS[dtype] * xl**power), color=INK2, fontsize=8,
                           xytext=(-4, -8), textcoords="offset points", ha="right", va="top")
            a.set_title(f"{dtype}: {title}", fontsize=9)
            if r == len(LOW) - 1:
                a.set_xlabel("κ(B_{A∖p}) at the step")
        if hidden:
            axes[r, 0].text(0.98, 0.02, f"{hidden} steps with κ > 1e17\n(singular after rounding) not shown",
                            transform=axes[r, 0].transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
    leg = axes[0, 0].legend(markerscale=3, title="implementation")
    for h in leg.legend_handles:
        h.set_alpha(1)
    _save(fig, figdir / "step_error_vs_kappa.png")


def step_error_vs_step(diags, figdir):
    ns = sorted({int(k[0].split("_n")[1].split("_")[0]) for k in diags})
    if not ns:
        return
    n = ns[-1]
    picks = [config_key("conditioned", n, {"kappa": 1e4}), config_key("collinear", n, {"eta": 1e-5}),
             config_key("harshaw", n, {"d": 50, "phi": 0.01, "corr": True})]
    cols = [("err_Bu", "‖B(û − u⁽⁶⁴⁾)‖₂"), ("err_u", "‖û − u⁽⁶⁴⁾‖∞"), ("err_p", "|p̂ − p⁽⁶⁴⁾|")]
    fig, axes = plt.subplots(len(picks), 3, figsize=(12, 3.2 * len(picks)), squeeze=False)
    for r, key in enumerate(picks):
        for dtype in DTYPES:
            d = diags.get((key, "lstsq", dtype))
            if d is None:
                continue
            for c, (f, title) in enumerate(cols):
                ok = np.isfinite(d[f]) & (d[f] > 0)
                axes[r, c].scatter(d["t"][ok], d[f][ok], s=5, alpha=0.4, lw=0,
                                   color=DTYPE_COLOR[dtype], label=dtype)
                axes[r, c].set_yscale("log")
                axes[r, c].set_title(f"lstsq, {key}: {title}", fontsize=8)
        axes[r, 0].set_ylabel("error")
    for c in range(3):
        axes[-1, c].set_xlabel("step t")
    leg = axes[0, 0].legend(markerscale=3)
    for h in leg.legend_handles:
        h.set_alpha(1)
    _save(fig, figdir / "step_error_vs_step.png")


def factor_drift(diags, figdir):
    keys = sorted({k[0] for k in diags if k[0].startswith("harshaw")})
    if not keys:
        return
    n = max(int(k.split("_n")[1].split("_")[0]) for k in keys)
    series = [("chol", "drift", "chol: ‖UᵀU − M‖_F/‖M‖_F"),
              ("chol_rk10", "drift", "chol, refactor every 10"),
              ("gs", "inv_drift", "gs: ‖CQ − I‖_F/√k")]
    colors = dict(zip([s[0] for s in series], DTYPE_COLOR.values()))
    for phi in (0.5, 0.1, 0.01):
        panels = [(d, c) for d in (5, 50) for c in (False, True)]
        fig, axes = plt.subplots(len(LOW), len(panels), figsize=(3.6 * len(panels), 6.5),
                                 squeeze=False, sharey="row")
        for r, dtype in enumerate(LOW):
            for c, (dd, corr) in enumerate(panels):
                key = config_key("harshaw", n, {"d": dd, "phi": phi, "corr": corr})
                a = axes[r, c]
                for impl, field, label in series:
                    d = diags.get((key, impl, dtype))
                    if d is None:
                        continue
                    y = np.where(d[field] > 0, d[field], np.nan)
                    for i, run in enumerate(np.unique(d["run"])):
                        m = d["run"] == run
                        a.plot(d["t"][m], y[m], color=colors[impl], lw=1, alpha=0.6,
                               label=label if i == 0 else None)
                a.set_yscale("log")
                a.set_title(f"{dtype}, d={dd}, {'corr. X' if corr else 'Gaussian X'}", fontsize=9)
                if r == len(LOW) - 1:
                    a.set_xlabel("step t")
            axes[r, 0].set_ylabel("relative factor error")
        axes[0, 0].legend()
        fig.suptitle(f"Factor drift, Harshaw design, n={n}, φ={phi:g}", color=INK, fontsize=10)
        _save(fig, figdir / f"factor_drift_phi{phi:g}.png")


def make_all(outdir):
    _style()
    outdir = Path(outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    rows = report.build_table(outdir)
    sweep_plots(rows, figdir)
    ccdf_plots(outdir, figdir)
    diags = _load_diags(outdir)
    step_error_vs_kappa(diags, figdir)
    step_error_vs_step(diags, figdir)
    factor_drift(diags, figdir)
    print(f"figures written to {figdir}")
