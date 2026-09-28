"""Norm distributions of Bz, Mahalanobis-whitened, for any --matrix B.

Each config's samples are WHITENED by their empirical m x m covariance, so
under joint Gaussianity the whitened coords are iid N(0,1) at every
precision and the norm distributions have exact references:

    ||W Bz||_inf : survival 1 - (1 - 2 Phi_bar(t))^m   (max of m iid N(0,1))
    ||W Bz||_2   : survival chi.sf(t, m)               (chi_m)

On (t^2, log-survival) axes the chi_m tail is a straight line. Deviations =
departures from joint Gaussianity (e.g. bounded support, discreteness),
NOT sigma inflation, which whitening removes.

Usage: python norm_dists.py [--matrix higgs --n 500]
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
from scipy import stats

import _bootstrap  # noqa: F401  # repo root on sys.path
from core import matrices, rollouts
from core.paths import figure_path


def whiten(bz: np.ndarray) -> np.ndarray:
    """Mahalanobis-whiten columns of bz (m, S): returns (m, S) with cov ~ I."""
    S = np.cov(bz)
    evals, evecs = np.linalg.eigh(S)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return W @ bz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    matrices.add_matrix_arguments(parser)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--sig-bits", type=int, nargs="+", default=[2, 12, 52])
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None)
    rollouts.add_noise_arguments(parser)
    args = parser.parse_args()

    sig_list = args.sig_bits
    num_samples = args.num_samples
    save_path = rollouts.output_path(
        args.save or figure_path("norm_dists", f"norm_dists_{args.matrix}.png"),
        args.noise_std)
    metadata = rollouts.experiment_metadata(
        args.noise_std, seed=args.seed, matrix=args.matrix, matrix_n=args.n,
        num_samples=num_samples, sig_bits=sig_list)
    B = matrices.matrix_family(args.matrix, m=args.m, n_max=args.n,
                               seed=args.seed)(args.n)
    m, n = B.shape

    data = {}
    for sb in sig_list:
        _, _, bz, _ = rollouts.run_samples(
            B, np.eye(m), num_samples,
            sig_bits=None if sb == 52 else sb,
            noise_std=args.noise_std, workers=args.workers, seed=args.seed,
        )
        data[sb] = whiten(bz)
        print(f"{sb}b done")

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sig_list)))
    fig, (ax_sc, ax_inf, ax_2) = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(
        f"Whitened norms of Bz — {args.matrix} B ({m}x{n}), {num_samples} walks\n"
        "(under joint Gaussianity whitened coords are iid N(0,1))"
    )

    # scatter of the first two whitened coords + unit circle: isotropy check
    for sb, col in zip(sig_list, colors):
        z = data[sb]
        ax_sc.plot(z[0], z[1], ".", color=col, alpha=0.08,
                   markersize=3, label=f"{sb}b")
    th = np.linspace(0, 2 * np.pi, 200)
    ax_sc.plot(np.cos(th), np.sin(th), "k--", lw=0.8)
    ax_sc.set_aspect("equal")
    ax_sc.set_xlabel("whitened Bz_0")
    ax_sc.set_ylabel("whitened Bz_1")
    ax_sc.set_title("whitened scatter, first 2 coords (unit circle)")
    ax_sc.legend(title="mantissa bits", markerscale=8)

    # ||W Bz||_inf survival vs max-of-m-iid reference
    for sb, col in zip(sig_list, colors):
        t = np.sort(np.abs(data[sb]).max(axis=0))
        surv = np.arange(len(t), 0, -1) / len(t)
        ax_inf.plot(t ** 2, surv, color=col, label=f"{sb}b")
    tt = np.linspace(0, np.sqrt(2 * np.log(max(m, 2))) + 2.0, 200)
    ref = 1.0 - (1.0 - 2.0 * stats.norm.sf(tt)) ** m   # max of m iid |N(0,1)|
    ax_inf.plot(tt ** 2, ref, "k--", lw=1, label=f"max of {m} iid N(0,1)")
    ax_inf.set_yscale("log")
    ax_inf.set_xlabel("t²")
    ax_inf.set_ylabel("Pr[‖W·Bz‖∞ > t]")
    ax_inf.set_title("‖·‖∞ tail (dashed: iid-Gaussian max)")
    ax_inf.legend(title="mantissa bits")

    # ||W Bz||_2 survival vs chi_m reference
    for sb, col in zip(sig_list, colors):
        t = np.sort(np.linalg.norm(data[sb], axis=0))
        surv = np.arange(len(t), 0, -1) / len(t)
        ax_2.plot(t ** 2, surv, color=col, label=f"{sb}b")
    ax_2.plot(tt ** 2, stats.chi.sf(tt, m), "k--", lw=1,
              label=f"χ_{m} tail")
    ax_2.set_yscale("log")
    ax_2.set_xlabel("t²")
    ax_2.set_ylabel("Pr[‖W·Bz‖₂ > t]")
    ax_2.set_title("‖·‖₂ tail (dashed: χ_m)")
    ax_2.legend(title="mantissa bits")

    fig.tight_layout()
    rollouts.save_figure(fig, save_path, metadata)
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
