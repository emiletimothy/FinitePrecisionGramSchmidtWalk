"""Phase diagram: log sigma-hat over the (n, mantissa-bits) plane — Higgs B.

Denser bit grid (2..16 plus 23, 52) x n (50..1600). Cells already in a
provenance-checked n_subgauss cache (the default noise-labelled path or
--rounding-cache PATH) are reused; the rest are computed here.

Overlays the empirical boundary n_crit ~ 10 * 2^b (where the rounding term
0.08·n·2^-b crosses the fp64 plateau sigma_0 ~ 0.75) and fp-format rows.

Usage: python phase_heatmap.py [--plot-only]
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
from n_subgauss import _directions_for, collect, FMT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--dirs", type=int, default=64)
    p.add_argument("--t", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=6)
    matrices.add_matrix_arguments(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--save", default=None)
    p.add_argument("--rounding-cache", default=None,
                   help="optional provenance-checked n_subgauss cache to reuse")
    rollouts.add_noise_arguments(p)
    args = p.parse_args()

    n_values = [50, 100, 200, 400, 800, 1600]
    all_bits = sorted(set(range(2, 17)) | {23, 52})
    rng = np.random.default_rng(args.seed)

    grid = np.full((len(all_bits), len(n_values)), np.nan)
    save_path = rollouts.output_path(
        args.save or figure_path("phase_heatmap", f"phase_heatmap_{args.matrix}.png"),
        args.noise_std)
    cache = save_path.replace(".png", "_cache.npz")
    settings = dict(noise_std=args.noise_std, seed=args.seed, matrix=args.matrix, matrix_n=max(n_values),
                    num_samples=args.num_samples, num_dirs=args.dirs, t=args.t)
    metadata = rollouts.experiment_metadata(**settings, n_values=n_values, sig_bits=all_bits)

    # reuse what we already measured
    if args.rounding_cache and not args.plot_only:
        z = rollouts.load_cache(args.rounding_cache, **settings)
        metadata["source_cache_metadata"] = rollouts.metadata_from_cache(z)
        for i, b in enumerate(z["sig_bits"]):
            if int(b) in all_bits:
                for j, n in enumerate(z["n_values"]):
                    if int(n) in n_values:
                        grid[all_bits.index(int(b)), n_values.index(int(n))] = z["sig_max"][i, j]

    if args.plot_only:
        zz = rollouts.load_cache(cache, **settings, n_values=n_values, sig_bits=all_bits)
        metadata = rollouts.metadata_from_cache(zz)
        grid = zz["grid"]
    else:
        missing = [(b, n) for b in all_bits for j, n in enumerate(n_values)
                   if np.isnan(grid[all_bits.index(b), j])]
        print(f"{len(missing)} cells to compute")
        get_B = matrices.matrix_family(args.matrix, m=args.m,
                                       n_max=max(n_values), seed=args.seed)
        by_n = {}
        for b, n in tqdm(missing, desc="heatmap cells"):
            if n not in by_n:
                by_n[n] = get_B(n)
            B = by_n[n]
            dirs = _directions_for(B, args.dirs, rng)
            _, projections, _, _ = rollouts.run_samples(
                B, dirs, args.num_samples,
                sig_bits=None if b == 52 else b,
                noise_std=args.noise_std, workers=args.workers, seed=args.seed,
            )
            grid[all_bits.index(b), n_values.index(n)] = \
                collect(projections, args.t)["sig_max"]
        rollouts.save_cache(cache, metadata, grid=grid, sig_bits=np.array(all_bits),
                 n_values=np.array(n_values))
        print(f"cached grid -> {cache}")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    # categorical rows: every bit-width gets equal height (the fp32/fp64 rows
    # would otherwise eat half a linear axis while staying flat)
    pc = ax.pcolormesh(np.arange(len(n_values) + 1),
                       np.arange(len(all_bits) + 1), np.log10(grid),
                       cmap="inferno", shading="auto",
                       vmin=-0.3, vmax=1.6)
    fig.colorbar(pc, ax=ax, label="log₁₀ σ̂ (worst direction)")

    ax.set_xticks(np.arange(len(n_values)) + 0.5)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_yticks(np.arange(len(all_bits)) + 0.5)
    ax.set_yticklabels([str(b) for b in all_bits], fontsize=7)
    for b, name in FMT.items():
        if b in all_bits:
            i = all_bits.index(b)
            ax.axhline(i + 1, color="white", lw=0.4, alpha=0.5)
            ax.text(len(n_values) + 0.15, i + 0.5, name, color="white",
                    va="center", fontsize=7)

    # empirical boundary n = 10·2^b  <=>  row index of b = log2(n/10),
    # drawn in categorical coordinates by interpolation
    ns = np.logspace(np.log10(min(n_values)), np.log10(max(n_values)), 200)
    bs = np.log2(ns / 10.0)
    row_of = np.interp(bs, all_bits, np.arange(len(all_bits)) + 0.5)
    col_of = np.interp(np.log10(ns), np.log10(n_values),
                       np.arange(len(n_values)) + 0.5)
    ok = (bs >= min(all_bits)) & (bs <= max(all_bits))
    ax.plot(col_of[ok], row_of[ok], "c--", lw=1.2,
            label="n = 10·2^b  (knee)")

    ax.set_xlabel("n")
    ax.set_ylabel("mantissa bits (sig_bits)")
    ax.set_title(f"σ̂ over the (n, precision) plane — {args.matrix} B, "
                 f"{args.num_samples} walks")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    rollouts.save_figure(fig, save_path, metadata)
    plt.close(fig)
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
