"""Does d·Bz stay Gaussian-shaped as sigma inflates with n?

At fixed low precision (fp8 E5M2 / E4M3), plot the standardised survival of
the worst direction |d*·Bz|/sigma-hat on (t^2, log-y) axes for increasing n.
If the curves collapse onto the N(0,1) tail (slope -1/2 straight line),
rounding inflates the SCALE but not the SHAPE of the output distribution.

Usage: python shape_persist.py
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
from n_subgauss import _directions_for, _subgaussian_sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    matrices.add_matrix_arguments(parser)
    rollouts.add_noise_arguments(parser)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    num_samples, num_dirs, workers, seed = 1000, 64, 6, 0
    save_path = rollouts.output_path(
        args.save or figure_path("shape_persist", f"shape_persist_{args.matrix}.png"),
        args.noise_std)
    metadata = rollouts.experiment_metadata(args.noise_std, seed=seed,
                                           matrix=args.matrix, matrix_n=1600,
                                           num_samples=num_samples, num_dirs=num_dirs,
                                           n_values=[100, 400, 1600], sig_bits=[2, 3])
    rng = np.random.default_rng(seed)
    n_values = [100, 400, 1600]
    sig_list = [2, 3]
    get_B = matrices.matrix_family(args.matrix, m=args.m,
                                   n_max=max(n_values), seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fig.suptitle("Worst-direction tail shape vs n — standardised by σ̂ "
                 f"({args.matrix} B, {num_samples} walks)")

    for ax, sb in zip(axes, sig_list):
        chop_label = "fp8-E5M2" if sb == 2 else "fp8-E4M3"
        cmap = plt.cm.copper(np.linspace(0.2, 0.8, len(n_values)))
        for n, col in zip(n_values, cmap):
            B = get_B(n)
            dirs = _directions_for(B, num_dirs, rng)
            _, projections, _, _ = rollouts.run_samples(
                B, dirs, num_samples, sig_bits=sb,
                noise_std=args.noise_std, workers=workers, seed=seed,
            )
            sig_d = np.array([_subgaussian_sigma(p)[0] for p in projections])
            j = int(np.argmax(sig_d))
            X = projections[j] / sig_d[j]
            t = np.sort(np.abs(X))
            surv = np.arange(len(t), 0, -1) / len(t)
            ax.plot(t ** 2, surv, color=col, lw=1.2,
                    label=f"n={n}, σ̂={sig_d[j]:.1f}")
        tt = np.linspace(0, np.sqrt(15), 50)
        ax.plot(tt ** 2, 2 * np.exp(-tt ** 2 / 2), "k--", lw=1,
                label="N(0,1) tail")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 3)
        ax.set_xlabel("t²")
        ax.set_title(f"{chop_label} ({sb}b)")
        ax.legend(fontsize=8)
        tqdm.write(f"done {sb}b")
    axes[0].set_ylabel("Pr[|d*·Bz|/σ̂ > t]")

    fig.tight_layout()
    rollouts.save_figure(fig, save_path, metadata)
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
