"""Subgaussianity vs mantissa bits at fixed n, for any --matrix B.

x = sig_bits, y = log sigma-hat. Prediction: sigma-hat ~ max(sigma_0,
c * n * 2^-bits), i.e. a plateau at the fp64 sigma_0 ~ 0.7 and a straight
descending line (slope -log2 per bit) below the transition bit.

Panels:
  A) worst-direction sigma-hat vs bits: moment estimate (solid) and psi_2
     Orlicz norm (dotted). Reference lines: sigma = 1 and the fitted
     c * n * 2^-b guide through the lowest bits.
  B) median sigma-hat over directions with a 10-90% band — uniformity.

m = 2, so directions are the full unit circle (64 angles). Reuses the
estimators from n_subgauss.py.

Usage:
    python bits_subgauss.py
    python bits_subgauss.py --n 500 --sig-bits 2 3 4 5 6 8 10 12 16 20 30 52
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
from n_subgauss import _directions_for, collect


def bits_subgauss(
    matrix: str = "higgs",
    m: int = 30,
    n: int = 500,
    sig_bits_values: list[int] | None = None,
    num_samples: int = 1000,
    num_dirs: int = 64,
    t: float = 3.0,
    workers: int | None = None,
    seed: int = 0,
    save_path: str | None = None,
    plot_only: bool = False,
    noise_std: float = 0.0,
) -> None:
    if sig_bits_values is None:
        sig_bits_values = list(range(2, 21)) + [30, 40, 52]
    if save_path is None:
        save_path = figure_path("bits_subgauss", f"bits_subgauss_{matrix}.png")

    save_path = rollouts.output_path(save_path, noise_std)
    cache_path = save_path.replace(".png", "_cache.npz")

    rng = np.random.default_rng(seed)
    if not plot_only:
        B = matrices.matrix_family(matrix, m=m, n_max=n, seed=seed)(n)
        m = B.shape[0]
    settings = dict(noise_std=noise_std, seed=seed, matrix=matrix, matrix_n=n,
                    num_samples=num_samples, num_dirs=num_dirs, t=t,
                    sig_bits=list(sig_bits_values))
    if not plot_only:
        settings["m"] = m
    metadata = rollouts.experiment_metadata(**settings)
    if plot_only:
        z = rollouts.load_cache(cache_path, **settings)
        metadata = rollouts.metadata_from_cache(z)
        # caches written before m was recorded fall back to the family's own m
        m = int(metadata.get("m", matrices.family_rows(matrix, m, n)))
        sig_bits_values = [int(b) for b in z["sig_bits"]]
        stats = {int(b): {k: float(z[k][i]) for k in z if k not in ("sig_bits", "metadata")}
                 for i, b in enumerate(sig_bits_values)}
    else:
        directions = _directions_for(B, num_dirs, rng)

        stats = {}
        for sig_bits in tqdm(sig_bits_values, desc="sig_bits"):
            _, projections, _, _ = rollouts.run_samples(
                B, directions, num_samples,
                sig_bits=None if sig_bits == 52 else sig_bits,
                noise_std=noise_std, workers=workers, seed=seed,
            )
            stats[sig_bits] = collect(projections, t)

        keys = list(stats[sig_bits_values[0]].keys())
        rollouts.save_cache(cache_path, metadata, sig_bits=np.array(sig_bits_values),
                 **{k: np.array([stats[b][k] for b in sig_bits_values])
                    for k in keys})
        print(f"cached stats -> {cache_path}")

    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(f"Subgaussianity vs mantissa bits — {matrix} B ({m}x{n}), "
                 f"{num_samples} walks")

    sig_max = [stats[b]["sig_max"] for b in sig_bits_values]
    psi2_max = [stats[b]["psi2_max"] for b in sig_bits_values]
    errs = [stats[b]["sig_max_err"] for b in sig_bits_values]

    ax_s.errorbar(sig_bits_values, sig_max, yerr=errs, marker="o",
                  markersize=4, label="moment est (worst dir)")
    ax_s.plot(sig_bits_values, psi2_max, ":", marker="x", markersize=4,
              label="ψ₂ norm (worst dir)")
    ax_s.axhline(1.0, color="gray", linestyle="--", lw=0.8, label="σ = 1")
    # model guides through the two lowest bit widths:
    #   1 + c·2^-b           (floor pinned at 1)
    #   σ₀ + c·2^-b          (additive error on a free plateau σ₀)
    #   √(σ₀² + (c·2^-b)²)   (independent error adds in quadrature)
    b0 = np.array(sig_bits_values[:2], dtype=float)
    s0v = np.array(sig_max[:2])
    hi = np.array(sig_bits_values) >= 10
    sig0 = float(np.median(np.array(sig_max)[hi]))
    c1 = float(np.mean((s0v - 1.0) * 2.0 ** b0))
    c2 = float(np.mean((s0v - sig0) * 2.0 ** b0))
    c3 = float(np.sqrt(np.mean((s0v ** 2 - sig0 ** 2) * 4.0 ** b0)))
    xs = np.linspace(min(sig_bits_values), max(sig_bits_values), 200)
    ax_s.plot(xs, 1.0 + c1 * 2.0 ** (-xs), "r--", lw=0.9, alpha=0.7,
              label=f"1 + {c1:.0f}·2⁻ᵇ")
    ax_s.plot(xs, sig0 + c2 * 2.0 ** (-xs), "--", color="seagreen", lw=0.9,
              alpha=0.7, label=f"σ₀ + {c2:.0f}·2⁻ᵇ")
    ax_s.plot(xs, np.sqrt(sig0 ** 2 + (c3 * 2.0 ** (-xs)) ** 2), "--",
              color="purple", lw=0.9, alpha=0.7,
              label=f"√(σ₀² + ({c3:.0f}·2⁻ᵇ)²)")
    print(f"fits: sig0={sig0:.3f}  c(1+c·2^-b)={c1:.1f}  "
          f"c(σ0+c·2^-b)={c2:.1f}  c(quadrature)={c3:.1f}")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("mantissa bits (sig_bits)")
    ax_s.set_ylabel("σ̂ (worst direction)")
    ax_s.set_title("σ̂ vs bits (log-y: slope −1/bit = ∝2⁻ᵇ)")
    ax_s.legend(fontsize=8)

    med = [stats[b]["sig_med"] for b in sig_bits_values]
    p10 = [stats[b]["sig_p10"] for b in sig_bits_values]
    p90 = [stats[b]["sig_p90"] for b in sig_bits_values]
    ax_d.fill_between(sig_bits_values, p10, p90, alpha=0.25)
    ax_d.plot(sig_bits_values, med, marker="s", markersize=4,
              label="median σ̂")
    ax_d.plot(sig_bits_values, sig_max, "--", lw=0.8, color="gray",
              label="max σ̂")
    ax_d.axhline(1.0, color="gray", linestyle="--", lw=0.8)
    ax_d.set_yscale("log")
    ax_d.set_xlabel("mantissa bits (sig_bits)")
    ax_d.set_ylabel("σ̂ over directions")
    ax_d.set_title("direction spread (band: 10–90%)")
    ax_d.legend(fontsize=8)

    fig.tight_layout()
    rollouts.save_figure(fig, save_path, metadata)
    plt.close(fig)
    print(f"saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subgaussianity vs mantissa bits at fixed n")
    matrices.add_matrix_arguments(parser)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--sig-bits", type=int, nargs="+", default=None,
                        help="(default: 2..20, 30, 40, 52)")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--dirs", type=int, default=64)
    parser.add_argument("--t", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true",
                        help="re-plot from the cached stats .npz, no rollouts")
    rollouts.add_noise_arguments(parser)
    args = parser.parse_args()

    bits_subgauss(
        matrix=args.matrix, m=args.m, n=args.n,
        sig_bits_values=args.sig_bits,
        num_samples=args.num_samples, num_dirs=args.dirs, t=args.t,
        workers=args.workers, seed=args.seed, save_path=args.save,
        plot_only=args.plot_only, noise_std=args.noise_std,
    )
