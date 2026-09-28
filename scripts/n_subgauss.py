"""Subgaussianity vs n — is the O(1)-subgaussian guarantee n-independent?

Three panels, x = n on every axis:

  A) worst-direction sigma-hat: moment estimate (solid) and the psi_2 Orlicz
     norm (dotted). Flat at sigma <= 1 means the guarantee holds; on log-log
     axes a polynomial growth reads directly as a slope (the adversarial
     lower-bound construction would give slope 2, sigma ~ a * n^2).
  B) Pr[|d* . Bz| > t] at a FIXED threshold t for the worst direction d* —
     an estimator-free survival count. Flat and <= 2 e^{-t^2/2} = pass.
  C) median sigma-hat over directions with a 10-90% band — shows whether
     any inflation is uniform across directions or concentrated in a few.

For m = 2 matrices (e.g. the two-feature Higgs B) the direction set is the
full unit circle (NUM_DIRS angles) — complete coverage, no sampling gaps.
For m > 2, directions are fixed random unit vectors + top SVD directions
(the direction set is held fixed in m, so coverage is consistent across n).

The B matrices are nested across n (prefix of one big column set), so the
n-dependence isn't confounded by a redrawn matrix.

Usage:
    python n_subgauss.py                          # Higgs 2x500-prefix B
    python n_subgauss.py --matrix lower_bound     # Theorem-4.1 matrix
    python n_subgauss.py --matrix clustered --m 30
    python n_subgauss.py --n 50 100 200 400 800 --sig-bits 2 6 12 52
"""

import os
os.environ.setdefault("chop_backend", "numpy")
# single-threaded BLAS so parallel rollout workers don't oversubscribe cores
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


def _subgaussian_sigma(vals: np.ndarray) -> tuple[float, float]:
    """
    Estimate the subgaussian parameter sigma two ways.

    Moment estimate: tightest lower bound across k=1,2,3 from
        sigma >= (E[X^{2k}] / (2k-1)!!)^{1/(2k)}

    Tail estimate: invert the tail bound P(|X|>t) <= 2 exp(-t^2/(2 sigma^2))
    at every empirical order statistic and take the max over the upper fifth.
    """
    N = len(vals)
    sigma_mom = max(
        np.mean(vals ** 2) ** 0.5,           # k=1, (2k-1)!! = 1
        (np.mean(vals ** 4) / 3) ** 0.25,    # k=2, (2k-1)!! = 3
        (np.mean(vals ** 6) / 15) ** (1/6),  # k=3, (2k-1)!! = 15
    )

    abs_sorted = np.sort(np.abs(vals))[::-1]          # descending order stats
    ranks = np.arange(1, N + 1, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_candidates = abs_sorted / np.sqrt(2 * np.log(2 * N / ranks))
    # only look at the upper tail (top 20%) to avoid log≈0 instability near median
    sigma_tail = float(np.nanmax(sigma_candidates[: max(1, N // 5)]))

    return sigma_mom, sigma_tail


def _psi2_norm(vals: np.ndarray) -> float:
    """Subgaussian Orlicz norm: inf{t : E[exp(X^2/t^2)] <= 2}, by bisection."""
    x = np.abs(np.asarray(vals, dtype=float))
    xmax = float(x.max())
    if xmax == 0.0:
        return 0.0
    n = len(x)

    def f(t):
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.mean(np.exp((x / t) ** 2))) - 2.0

    # even the single largest sample forces t >= xmax / sqrt(log(2n));
    # start a hair below so f(lo) > 0 for sure
    lo = xmax / np.sqrt(np.log(2 * n)) * 0.999999
    hi = max(lo * 4.0, 4.0 * float(x.std()) + 1e-12)
    while f(hi) > 0.0:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _directions_for(B: np.ndarray, num_dirs: int, rng: np.random.Generator):
    """Direction set in R^m: full circle when m == 2, else random + SVD."""
    m = B.shape[0]
    if m == 2:
        th = np.linspace(0.0, np.pi, num_dirs, endpoint=False)
        return np.stack([np.cos(th), np.sin(th)], axis=1)
    rand = rng.standard_normal((num_dirs, m))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(B, full_matrices=False)
    return np.vstack([rand, U[:, : min(10, U.shape[1])].T])


def collect(projections: np.ndarray, t: float, n_folds: int = 5):
    """Per-config summary stats from projections (num_dirs, num_samples).

    The worst direction is selected on the first half of the walks and all
    statistics are evaluated on the held-out second half, so the reported
    sigmas carry no selection bias. Sigmas/exceedance use centered samples;
    the bias (empirical mean) is reported separately.
    """
    sel, ev = np.array_split(projections, 2, axis=1)
    j = int(np.argmax([_subgaussian_sigma(p - p.mean())[0] for p in sel]))
    ev_c = ev - ev.mean(axis=1, keepdims=True)
    sig_mom = np.array([_subgaussian_sigma(p)[0] for p in ev_c])
    sig_psi2 = np.array([_psi2_norm(p) for p in ev_c])
    worst = ev_c[j]
    exc = float(np.mean(np.abs(worst) > t))
    fold_sigs = [
        max(_subgaussian_sigma(p)[0] for p in fold)
        for fold in np.array_split(ev_c, n_folds, axis=1)
        if fold.shape[1]
    ]
    return dict(
        sig_max=float(sig_mom[j]),
        sig_max_err=float(np.std(fold_sigs)) if fold_sigs else 0.0,
        psi2_max=float(sig_psi2[j]),
        sig_med=float(np.median(sig_mom)),
        sig_p10=float(np.percentile(sig_mom, 10)),
        sig_p90=float(np.percentile(sig_mom, 90)),
        exc=exc,
        bias=float(ev[j].mean()),
        max_bias=float(np.abs(ev.mean(axis=1)).max()),
    )


# mantissa bits -> fp format name (sig_bits counts fraction bits)
FMT = {2: "fp8-E5M2", 3: "fp8-E4M3", 7: "bf16", 10: "fp16", 23: "fp32", 52: "fp64"}


def _lab(b: int) -> str:
    return f"{FMT.get(b, str(b) + 'b')} ({b}b)" if b in FMT else f"{b}b"


def n_subgauss(
    matrix: str = "higgs",
    m: int = 30,
    n_values: list[int] | None = None,
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
    if n_values is None:
        n_values = [50, 100, 200, 400, 800, 1600]
    if sig_bits_values is None:
        sig_bits_values = [2, 3, 7, 10, 23, 52]
    if save_path is None:
        save_path = figure_path("n_subgauss", f"n_subgauss_{matrix}.png")
    save_path = rollouts.output_path(save_path, noise_std)
    cache_path = save_path.replace(".png", "_cache.npz")

    rng = np.random.default_rng(seed)
    if not plot_only:
        get_B = matrices.matrix_family(matrix, m=m, n_max=max(n_values), seed=seed)
        m = get_B(n_values[0]).shape[0]
    settings = dict(noise_std=noise_std, seed=seed, matrix=matrix,
                    matrix_n=max(n_values), num_samples=num_samples, num_dirs=num_dirs, t=t,
                    n_values=list(n_values), sig_bits=list(sig_bits_values))
    if not plot_only:
        settings["m"] = m
    metadata = rollouts.experiment_metadata(**settings)

    if plot_only:
        z = rollouts.load_cache(cache_path, **settings)
        metadata = rollouts.metadata_from_cache(z)
        m = int(metadata.get("m", m))
        sig_bits_values = [int(b) for b in z["sig_bits"]]
        n_values = [int(v) for v in z["n_values"]]
        stats = {}
        for i, b in enumerate(sig_bits_values):
            for j, v in enumerate(n_values):
                stats[(b, v)] = {k: float(z[k][i, j])
                                 for k in z if k not in ("sig_bits", "n_values", "metadata")}
    else:
        # get_B was built above: one nested column set, B(n) = n-column prefix.
        stats = {}
        for sig_bits in tqdm(sig_bits_values, desc="sig_bits"):
            for n in n_values:
                B = get_B(n)
                directions = _directions_for(B, num_dirs, rng)
                _, projections, _, _ = rollouts.run_samples(
                    B, directions, num_samples,
                    sig_bits=None if sig_bits == 52 else sig_bits,
                    noise_std=noise_std, workers=workers, seed=seed,
                )
                stats[(sig_bits, n)] = collect(projections, t)

        keys = list(stats[(sig_bits_values[0], n_values[0])].keys())
        rollouts.save_cache(cache_path, metadata, sig_bits=np.array(sig_bits_values),
                 n_values=np.array(n_values),
                 **{k: np.array([[stats[(b, v)][k] for v in n_values]
                                 for b in sig_bits_values])
                    for k in keys})
        print(f"cached stats -> {cache_path}")

    colors = plt.cm.viridis(np.linspace(0, 1, len(sig_bits_values)))
    fig, (ax_s, ax_e, ax_d) = plt.subplots(1, 3, figsize=(16, 4.2))
    fig.suptitle(f"Subgaussianity vs n — {matrix} B ({m}x.), {num_samples} walks")

    # A) worst-direction sigma-hat: moments solid, psi_2 dotted
    for sig_bits, col in zip(sig_bits_values, colors):
        s = [stats[(sig_bits, n)] for n in n_values]
        ax_s.errorbar(n_values, [x["sig_max"] for x in s],
                      yerr=[x["sig_max_err"] for x in s], marker="o",
                      markersize=4, color=col, label=_lab(sig_bits))
        ax_s.plot(n_values, [x["psi2_max"] for x in s], ":", color=col,
                  marker="x", markersize=4)
    ax_s.axhline(1.0, color="gray", linestyle="--", lw=0.8, label="σ = 1")
    # slope-1 guide: sigma ~ n, anchored at the lowest-precision first point
    s_lo = stats[(min(sig_bits_values), n_values[0])]["sig_max"]
    ax_s.plot(n_values, s_lo * np.array(n_values) / n_values[0], "k:",
              lw=0.8, alpha=0.6, label="∝ n")
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("n")
    ax_s.set_ylabel("σ̂ (worst direction)")
    ax_s.set_title("max σ̂ vs n (dotted: ψ₂ norm)")
    ax_s.legend(title="mantissa bits", fontsize=8)

    # B) fixed-threshold exceedance for the worst direction
    floor = 0.5 / num_samples
    for sig_bits, col in zip(sig_bits_values, colors):
        exc = [max(stats[(sig_bits, n)]["exc"], floor) for n in n_values]
        ax_e.plot(n_values, exc, marker="o", markersize=4, color=col,
                  label=_lab(sig_bits))
    bound = 2.0 * np.exp(-(t ** 2) / 2.0)
    ax_e.axhline(bound, color="gray", linestyle="--", lw=0.8,
                 label=f"2e^(-t²/2), t={t:g}")
    ax_e.axhline(floor, color="lightgray", linestyle=":", lw=0.8)
    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlabel("n")
    ax_e.set_ylabel(f"Pr[|d*·Bz| > {t:g}]")
    ax_e.set_title("worst-direction exceedance (flat ≤ bound = pass)")
    ax_e.legend(title="mantissa bits", fontsize=8)

    # C) median sigma-hat with 10-90% band over directions
    for sig_bits, col in zip(sig_bits_values, colors):
        s = [stats[(sig_bits, n)] for n in n_values]
        ax_d.fill_between(n_values, [x["sig_p10"] for x in s],
                          [x["sig_p90"] for x in s], color=col, alpha=0.2)
        ax_d.plot(n_values, [x["sig_med"] for x in s], marker="s",
                  markersize=4, color=col, label=_lab(sig_bits))
    ax_d.axhline(1.0, color="gray", linestyle="--", lw=0.8)
    ax_d.set_xscale("log")
    ax_d.set_xlabel("n")
    ax_d.set_ylabel("σ̂ over directions")
    ax_d.set_title("median σ̂ vs n (band: 10–90%)")
    ax_d.legend(title="mantissa bits", fontsize=8)

    fig.tight_layout()
    rollouts.save_figure(fig, save_path, metadata)
    plt.close(fig)
    print(f"saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subgaussianity vs n: worst-direction sigma, exceedance, spread"
    )
    matrices.add_matrix_arguments(parser)
    parser.add_argument("--n", type=int, nargs="+", default=None,
                        help="n values (default: 50 100 200 400 800 1600)")
    parser.add_argument("--sig-bits", type=int, nargs="+", default=None,
                        help="mantissa widths (default: 2 3 7 10 23 52 = "
                             "E5M2 E4M3 bf16 fp16 fp32 fp64)")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--dirs", type=int, default=64,
                        help="directions (m=2: circle samples; else random+SVD)")
    parser.add_argument("--t", type=float, default=3.0,
                        help="fixed exceedance threshold (default 3)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true",
                        help="re-plot from the cached stats .npz, no rollouts")
    rollouts.add_noise_arguments(parser)
    args = parser.parse_args()

    n_subgauss(
        matrix=args.matrix, m=args.m, n_values=args.n,
        sig_bits_values=args.sig_bits, num_samples=args.num_samples,
        num_dirs=args.dirs, t=args.t, workers=args.workers,
        seed=args.seed, save_path=args.save, plot_only=args.plot_only, noise_std=args.noise_std,
    )
