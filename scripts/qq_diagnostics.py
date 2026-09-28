"""Gaussianity diagnostics for Bz samples, for any --matrix B.

Per sig_bits:

  (a) Q-Q of each Bz coordinate, standardised by its own sigma-hat, against
      N(0,1) quantiles -- a straight diagonal means Gaussian;
  (b) survival of |coord| on (t^2, log-y) axes with the N(0,1) reference
      slope -- straight line of slope -1/2 means exactly Gaussian;
  (c) standardised density (log-y) vs the N(0,1) pdf.

For large m only the first few coordinates are drawn.

Usage: python qq_diagnostics.py [--matrix higgs --n 500]
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

MAX_COORDS = 4   # draw at most this many coordinates per panel


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
        args.save or figure_path("qq_diagnostics", f"qq_diagnostics_{args.matrix}.png"),
        args.noise_std)
    metadata = rollouts.experiment_metadata(
        args.noise_std, seed=args.seed, matrix=args.matrix, matrix_n=args.n,
        num_samples=num_samples, sig_bits=sig_list)
    B = matrices.matrix_family(args.matrix, m=args.m, n_max=args.n,
                               seed=args.seed)(args.n)
    m, n = B.shape
    n_coords = min(m, MAX_COORDS)

    data = {}
    for sb in sig_list:
        _, _, bz, _ = rollouts.run_samples(
            B, np.eye(m), num_samples,
            sig_bits=None if sb == 52 else sb,
            noise_std=args.noise_std, workers=args.workers, seed=args.seed,
        )
        data[sb] = bz                       # (m, num_samples)
        print(f"{sb}b done: std per coord = {bz.std(axis=1)}")

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sig_list)))
    fig, (ax_qq, ax_sv, ax_de) = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(f"Gaussianity check — {args.matrix} B ({m}x{n}), "
                 f"{num_samples} walks")

    # (a) standardised Q-Q vs N(0,1)
    q = stats.norm.ppf((np.arange(1, num_samples + 1) - 0.5) / num_samples)
    for sb, col in zip(sig_list, colors):
        for i in range(n_coords):
            s = np.sort(data[sb][i] / data[sb][i].std())
            ax_qq.plot(q, s, color=col, alpha=0.8,
                       label=f"{sb}b" if i == 0 else None)
    lim = max(abs(ax_qq.get_xlim()[0]), abs(ax_qq.get_ylim()[1]))
    ax_qq.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax_qq.set_xlabel("N(0,1) quantile")
    ax_qq.set_ylabel("standardised Bz coord quantile")
    ax_qq.set_title(f"Q-Q vs Gaussian (first {n_coords} coords)")
    ax_qq.legend(title="mantissa bits")

    # (b) survival of |standardised coord| on (t^2, log-y): N(0,1) = slope -1/2
    for sb, col in zip(sig_list, colors):
        for i in range(n_coords):
            t = np.sort(np.abs(data[sb][i] / data[sb][i].std()))
            surv = np.arange(len(t), 0, -1) / len(t)
            ax_sv.plot(t ** 2, surv, color=col, alpha=0.8,
                       label=f"{sb}b" if i == 0 else None)
    tt = np.linspace(0, ax_sv.get_xlim()[1], 10)
    ax_sv.plot(tt, 2 * np.exp(-tt / 2), "k--", lw=0.8, label="N(0,1) tail")
    ax_sv.set_yscale("log")
    ax_sv.set_xlabel("t²")
    ax_sv.set_ylabel("Pr[|Bz_i|/σ > t]")
    ax_sv.set_title("standardised tail (dashed: Gaussian)")
    ax_sv.legend(title="mantissa bits")

    # (c) standardised density vs N(0,1) pdf
    for sb, col in zip(sig_list, colors):
        z = (data[sb] / data[sb].std(axis=1, keepdims=True)).ravel()
        ax_de.hist(z, bins=120, density=True, histtype="step",
                   color=col, label=f"{sb}b")
    xs = np.linspace(-6, 6, 400)
    ax_de.plot(xs, stats.norm.pdf(xs), "k--", lw=1, label="N(0,1)")
    ax_de.set_yscale("log")
    ax_de.set_xlabel("Bz_i / σ")
    ax_de.set_title("standardised density (log-y)")
    ax_de.legend(title="mantissa bits")

    fig.tight_layout()
    rollouts.save_figure(fig, save_path, metadata)
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
