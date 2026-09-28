"""Rounding vs matched-magnitude additive noise — Higgs B, sigma-hat vs n.

For each fp format b, compares:
  * rounding:  chop = make_round(b)        (taken from n_subgauss cache)
  * noise:     additive per-step noise with std = 2^-b, fp64 arithmetic

Prediction: rounding error accumulates coherently (~n·2^-b) while independent
additive noise accumulates as a random walk (~sqrt(n)·2^-b) — different
exponents, so both should collapse on their own scaling variable.

Panels: (A) sigma-hat vs n, solid=rounding / dashed=noise;
        (B) rounding points vs n·2^-b (collapse check);
        (C) noise points vs sqrt(n)·2^-b (collapse check).

Usage: python noise_vs_chop.py [--plot-only]
"""

import os
os.environ.setdefault("chop_backend", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import _bootstrap  # noqa: F401  # repo root on sys.path
from core import matrices, rollouts
from core.paths import figure_path
from n_subgauss import _directions_for, collect, FMT, _lab


def main():
    p = argparse.ArgumentParser()
    matrices.add_matrix_arguments(p)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--dirs", type=int, default=64)
    p.add_argument("--t", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--rounding-cache", default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--noise-scale", type=rollouts.noise_std_value, default=1.0,
                   help="noise-arm standard deviation is this scale times 2^-b; 0 disables it")
    rollouts.add_noise_arguments(p, scope="the rounding arm only; --noise-scale controls the noise arm")
    args = p.parse_args()

    cache_round = args.rounding_cache or rollouts.output_path(
        figure_path("n_subgauss", f"n_subgauss_{args.matrix}.png"),
        args.noise_std).replace(".png", "_cache.npz")
    save_path = rollouts.output_path(
        args.save or figure_path("noise_vs_chop", f"noise_vs_chop_{args.matrix}.png"),
        args.noise_std).replace(
        ".png", f"_scale{args.noise_scale:.17g}.png")
    cache_noise = save_path.replace(".png", "_cache.npz")
    rng = np.random.default_rng(args.seed)
    settings = dict(noise_std=args.noise_std, seed=args.seed, matrix=args.matrix,
                    num_samples=args.num_samples, num_dirs=args.dirs, t=args.t)

    # rounding stats from the n_sweep cache
    z = rollouts.load_cache(cache_round, **settings)
    sig_bits_values = [int(b) for b in z["sig_bits"]]
    n_values = [int(v) for v in z["n_values"]]
    settings.update(matrix_n=max(n_values), comparison_noise_scale=args.noise_scale,
                    sig_bits=sig_bits_values, n_values=n_values)
    if rollouts.metadata_from_cache(z).get("matrix_n") != max(n_values):
        raise ValueError("Rounding cache matrix size does not match its n grid")
    metadata = rollouts.experiment_metadata(**settings,
                                           rounding_source=rollouts.metadata_from_cache(z))
    print(f"Noise-only arm: std={args.noise_scale:g} * 2^-b")
    stats_r = {(int(b), int(n)): {"sig_max": float(z["sig_max"][i, j])}
               for i, b in enumerate(sig_bits_values)
               for j, n in enumerate(n_values)}

    if args.plot_only:
        # Verify the noise-arm settings match the cache, but keep the fresh
        # metadata: the figure must attribute the rounding arm actually read
        # above, not the snapshot stored when the noise arm was generated.
        zn = rollouts.load_cache(cache_noise, **settings)
        stats_nz = {(int(b), int(n)): {"sig_max": float(zn["sig_max"][i, j])}
                    for i, b in enumerate(sig_bits_values)
                    for j, n in enumerate(n_values)}
    else:
        get_B = matrices.matrix_family(args.matrix, m=args.m,
                                       n_max=max(n_values), seed=args.seed)
        stats_nz = {}
        for sig_bits in tqdm(sig_bits_values, desc="noise sig_bits"):
            for n in n_values:
                B = get_B(n)
                dirs = _directions_for(B, args.dirs, rng)
                _, projections, _, _ = rollouts.run_samples(
                    B, dirs, args.num_samples,
                    sig_bits=None, noise_std=args.noise_scale * 2.0 ** (-sig_bits),
                    workers=args.workers, seed=args.seed,
                )
                stats_nz[(sig_bits, n)] = collect(projections, args.t)
        rollouts.save_cache(cache_noise, metadata,
                 **{k: np.array([[stats_nz[(b, n)][k] for n in n_values]
                                 for b in sig_bits_values])
                    for k in stats_nz[sig_bits_values[0], n_values[0]]})
        print(f"cached noise stats -> {cache_noise}")

    colors = plt.cm.viridis(np.linspace(0, 1, len(sig_bits_values)))
    fig, (ax_s, ax_cr, ax_cn) = plt.subplots(1, 3, figsize=(16, 4.2))
    fig.suptitle(f"Rounding vs matched additive noise — {args.matrix} B, "
                 f"{args.num_samples} walks")

    for b, col in zip(sig_bits_values, colors):
        sr = [stats_r[(b, n)]["sig_max"] for n in n_values]
        sn = [stats_nz[(b, n)]["sig_max"] for n in n_values]
        ax_s.plot(n_values, sr, "o-", markersize=4, color=col,
                  label=_lab(b))
        ax_s.plot(n_values, sn, "x--", markersize=4, color=col, alpha=0.7)
        ax_cr.plot(np.array(n_values) * 2.0 ** (-b), sr, "o", markersize=4,
                   color=col, label=_lab(b))
        ax_cn.plot(np.sqrt(n_values) * 2.0 ** (-b), sn, "o", markersize=4,
                   color=col, label=_lab(b))

    ax_s.plot([], [], "k-", label="rounding")
    ax_s.plot([], [], "k--", label="additive noise")
    ax_s.axhline(1.0, color="gray", linestyle="--", lw=0.8)
    ax_s.set_xscale("log"); ax_s.set_yscale("log")
    ax_s.set_xlabel("n"); ax_s.set_ylabel("σ̂ (worst direction)")
    ax_s.set_title("σ̂ vs n — solid: rounding, dashed: noise")
    ax_s.legend(fontsize=6.5, ncol=2)

    xs = np.logspace(-6, 3, 200)
    ax_cr.plot(xs, np.sqrt(0.74 ** 2 + (0.08 * xs) ** 2), "k--", lw=0.8,
               label="√(0.74²+(0.08x)²)")
    ax_cr.axhline(1.0, color="gray", linestyle="--", lw=0.8)
    ax_cr.set_xscale("log"); ax_cr.set_yscale("log")
    ax_cr.set_xlabel("n · 2^{-bits}")
    ax_cr.set_title("rounding collapse on n·ε")
    ax_cr.legend(fontsize=7)

    ax_cn.axhline(1.0, color="gray", linestyle="--", lw=0.8)
    ax_cn.set_xscale("log"); ax_cn.set_yscale("log")
    ax_cn.set_xlabel("√n · 2^{-bits}")
    ax_cn.set_title("noise collapse on √n·ε")
    ax_cn.legend(fontsize=7)

    fig.tight_layout()
    rollouts.save_figure(fig, save_path, metadata)
    plt.close(fig)
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
