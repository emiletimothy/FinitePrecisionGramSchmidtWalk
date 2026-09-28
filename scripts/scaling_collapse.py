"""Scaling collapse: σ̂ vs n·2^-b (natural matrix) and a·n² (lower bound).

Composite of two existing stats caches — no new rollouts:

  left : n_subgauss cache for --matrix (default higgs), x = n·2^-b.
         Prediction: the coherent-accumulation model σ ~ max(σ₀, c·n·2^-b)
         collapses all (b, n) cells onto one slope-1 line.
  right: lb_n_subgauss cache (adversarial Theorem-4.1 matrix), x = a·n²
         with a = 2^-b. Prediction: the walk saturates the Ω(a·n²) lower
         bound — again a slope-1 line. Open markers flag cells outside the
         theorem's precondition a < 1/(8n).

Both panels also draw σ = 1 (the subgaussian guarantee of the exact
algorithm) and annotate the fitted log-log slope, which should be ≈ 1 in
the degraded regime if the collapse variable is right.

Usage:
    python scripts/scaling_collapse.py                     # from default caches
    python scripts/scaling_collapse.py --matrix clustered  # other n_subgauss cache
    python scripts/scaling_collapse.py --n-cache P --lb-cache P --save out.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401  # repo root on sys.path
from core import rollouts
from core.paths import figure_path
from n_subgauss import FMT, _lab
from lb_subgauss import _valid_for_adversarial


def _collapse_panel(ax, sig_bits, n_values, sig_max, bias_max, power, model,
                    validity=None):
    """loglog of the worst-direction *uncentered* σ̂ vs n**power · 2^-b.

    The uncentered scale sqrt(sig_c² + bias²) is what the discrepancy bounds
    measure: for the adversarial matrix the deterministic per-step error
    lives in E[d·Bz] (the centered σ̂ alone stays O(1)), so bias must be
    folded back in — on natural matrices bias ≪ σ̂ and this is a no-op.

    model="quad":   dashed σ = sqrt(σ₀² + (c·x)²), σ₀ = fp64 plateau,
                    c = median sqrt((y²−σ₀²)/x²) over the degraded regime.
    model="linear": dashed k·x/√2, k = median(y·√2/x) over degraded regime.
    Returns (σ₀ or k, label of the fitted curve).
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(sig_bits)))
    xs, ys, ns = [], [], []
    for i, b in enumerate(sig_bits):
        b = int(b)
        x = np.asarray(n_values, dtype=float) ** power * 2.0 ** -b
        y = np.hypot(np.asarray(sig_max[i], float),
                     np.asarray(bias_max[i], float)) if bias_max is not None \
            else np.asarray(sig_max[i], dtype=float)
        if validity is not None:
            bad = np.array([not validity(int(n), b) for n in n_values])
            ax.loglog(x[bad], y[bad], "o", mfc="none", mec=colors[i], ms=5)
        ax.loglog(x, y, "o", ms=5, color=colors[i], label=_lab(b))
        xs.append(x)
        ys.append(y)
        ns.append(np.asarray(n_values, dtype=float))
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    ns = np.concatenate(ns)

    # plateau = highest-precision row available
    j = int(np.argmax(sig_bits))
    sigma0 = float(np.median(
        np.hypot(sig_max[j], bias_max[j]) if bias_max is not None
        else sig_max[j]))
    deg = np.isfinite(ys) & (ys > 2.0 * sigma0) & (xs > 0)
    # the a·n² limb is only visible before the walk saturates: the e_0
    # discrepancy cannot exceed n/√2 (z in {-1,+1}^n), so cells near that
    # ceiling follow y = n/√2 instead of y ~ a·n² and are excluded from the fit
    if model == "linear":
        fit_mask = deg & (ys < 0.6 * ns / np.sqrt(2.0))
        if not fit_mask.any():
            fit_mask = deg
    else:
        fit_mask = deg

    # draw the fit only over the degraded regime: start a little left of the
    # point where the model crosses the plateau, so the line doesn't drag the
    # axes down through decades where it says nothing
    if model == "quad":
        c2 = np.nanmedian((ys[fit_mask] ** 2 - sigma0 ** 2) / xs[fit_mask] ** 2)
        c = np.sqrt(max(c2, 0.0))
        x0 = sigma0 / c if c > 0 else xs.max()
    else:  # linear: k·x/√2
        k = float(np.nanmedian(ys[fit_mask] * np.sqrt(2.0) / xs[fit_mask]))
        c = np.nan
        x0 = sigma0 * np.sqrt(2.0) / k if k > 0 else xs.max()

    xr = np.geomspace(max(x0 / 4.0, xs.min()), xs.max(), 100)
    if model == "quad":
        ax.loglog(xr, np.sqrt(sigma0 ** 2 + (c * xr) ** 2), "k--", lw=1.2,
                  label=f"√({sigma0:.2g}² + ({c:.2g}x)²)")
        fit = c
    else:
        ax.loglog(xr, k * xr / np.sqrt(2.0), "k--", lw=1.2,
                  label=f"{k:.2g}·x/√2")
        fit = k
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="σ = 1")
    if validity is not None:
        ax.loglog([], [], "o", mfc="none", mec="gray", ms=5,
                  label="a ≥ 1/(8n)")
    return fit


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", default="higgs",
                   help="matrix for the n·2^-b panel (default higgs)")
    p.add_argument("--n-cache", default=None,
                   help="n_subgauss stats cache (default: figures/n_subgauss/)")
    p.add_argument("--lb-cache", default=None,
                   help="lb_subgauss stats cache (default: figures/lb_subgauss/)")
    p.add_argument("--save", default=None)
    rollouts.add_noise_arguments(p)
    args = p.parse_args()

    n_cache = args.n_cache or rollouts.output_path(
        figure_path("n_subgauss", f"n_subgauss_{args.matrix}.png"),
        args.noise_std).replace(".png", "_cache.npz")
    lb_cache = args.lb_cache or figure_path("lb_subgauss", "lb_n_subgauss_cache.npz")
    save_path = rollouts.output_path(
        args.save or figure_path("scaling_collapse",
                                 f"scaling_collapse_{args.matrix}.png"),
        args.noise_std)

    for path, what in ((n_cache, "n_subgauss"), (lb_cache, "lb_subgauss")):
        try:
            open(path, "rb").close()
        except OSError:
            raise SystemExit(
                f"missing {what} cache at {path} — run "
                f"`python scripts/{what}.py` (matching --matrix/--noise-std) first")

    z = rollouts.load_cache(n_cache, noise_std=args.noise_std)
    sig_bits_n = [int(b) for b in z["sig_bits"]]
    n_vals = [int(v) for v in z["n_values"]]
    sig_max_n = np.asarray(z["sig_max"])
    bias_n = np.asarray(z["max_bias"]) if "max_bias" in z else \
        (np.asarray(z["bias"]) if "bias" in z else None)
    n_meta = rollouts.metadata_from_cache(z)

    with np.load(lb_cache) as lb:
        sig_bits_lb = [int(b) for b in lb["sig_bits"]]
        n_vals_lb = [int(v) for v in lb["n_values"]]
        sig_max_lb = np.asarray(lb["sig_max"])
        bias_lb = np.asarray(lb["max_bias"]) if "max_bias" in lb else None

    fig, (ax_n, ax_lb) = plt.subplots(1, 2, figsize=(13, 4.6))

    c_n = _collapse_panel(ax_n, sig_bits_n, n_vals, sig_max_n, bias_n,
                          power=1, model="quad")
    ax_n.set_xlabel("n · 2^{-bits}")
    ax_n.set_ylabel("σ̂ (worst dir)")
    ax_n.set_title(f"{args.matrix} B: collapse on n·ε")
    ax_n.legend(title="mantissa bits", fontsize=7, title_fontsize=8)

    k_lb = _collapse_panel(ax_lb, sig_bits_lb, n_vals_lb, sig_max_lb, bias_lb,
                           power=2, model="linear",
                           validity=_valid_for_adversarial)
    ax_lb.set_xlabel("n² · 2^{-bits}")
    ax_lb.set_title("lower-bound: collapse on n²·ε")
    ax_lb.legend(title="mantissa bits", fontsize=7, title_fontsize=8)

    fig.tight_layout()
    metadata = rollouts.experiment_metadata(
        args.noise_std, seed=0, matrix=args.matrix, kind="scaling_collapse",
        sources=dict(n_cache=n_meta.get("git_revision"),
                     lb_cache=lb_cache),
        fits=dict(n_coherent=c_n, lb_prefactor=k_lb))
    rollouts.save_figure(fig, save_path, metadata)
    plt.close(fig)
    print(f"fits: sqrt(σ₀²+(c·n·2^-b)²) c={c_n:.3g}; "
          f"lb |E[d·Bz]| ≈ {k_lb:.3g}·(n²·2^-b)/√2")
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
