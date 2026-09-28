"""Subgaussianity figure set for the Theorem-4.1 lower-bound matrix.

Same panels as n_subgauss.py / bits_subgauss.py, but for
    B in R^{n+1} x n with columns v_i = (e_0 + e_i)/sqrt(2)
under lower_bound_walk (adversarial per-step error a = 2^-sig_bits for the
first L = n/4 iterations). The e_0 direction is always included in the
direction set — it is the adversarial axis where the discrepancy is forced.

Cells violating the theorem's hypothesis a < 1/(8n) are drawn with open
markers on the n-sweep panels.

Usage:
    python lb_subgauss.py                  # both figures
    python lb_subgauss.py --plot-only      # re-render from caches
"""

import os
os.environ.setdefault("chop_backend", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import _bootstrap  # noqa: F401  # repo root on sys.path
from core import lpla
from core.matrices import lower_bound_matrix
from core.paths import figure_path
from n_subgauss import collect, FMT, _lab


def _valid_for_adversarial(n: int, sig_bits: int) -> bool:
    """Error a = 2^{-sig_bits} must satisfy a < 1/(8n) for the construction."""
    a = 2.0 ** (-sig_bits)
    return a * 8.0 * n < 1.0


def lower_bound_walk(B, a, chop=None, seed=None):
    """One GSW rollout on the lower-bound matrix with adversarial error a.

    The construction follows Theorem 4.1: for the first L = floor(n/4)
    iterations, every active non-pivot coordinate receives an additive
    error a; the pivot and frozen coordinates receive none.  This produces
    an e_0 discrepancy of Theta(a n^2) when a is small enough.
    """
    m, n = B.shape
    r = chop if chop is not None else lambda x: x
    B0 = np.asarray(B, dtype=float)
    B = r(B0.copy())
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    L = n // 4

    for t in range(n):
        active = np.where(np.abs(z) < 1 - 1e-9)[0]
        k = active.size
        if k == 0:
            break

        p = active[0]
        free = active[active != p]

        u = np.zeros(n)
        u[p] = 1.0
        u[free] = float(r(np.array(-1.0 / k)))
        u = r(u)

        # feasible interval for the active coordinates, exactly as in gsw.py
        z_a = z[active]
        u_a = u[active]
        r1 = r((-1.0 - z_a) / u_a)
        r2 = r((1.0 - z_a) / u_a)
        lo = r(np.minimum(r1, r2))
        hi = r(np.maximum(r1, r2))
        delta_min = float(r(np.array(np.max(lo))))
        delta_max = float(r(np.array(np.min(hi))))
        total = float(r(np.array(abs(delta_max) + abs(delta_min))))
        if total < 1e-15:
            break

        prob = float(r(np.array(abs(delta_min) / total)))
        delta = delta_max if rng.random() < prob else delta_min
        delta = float(r(np.array(delta)))

        # ideal GSW step
        z = r(z + r(delta * u))
        # deterministic adversarial error e_t for the first L iterations
        if t < L:
            z[free] = r(z[free] + a)

        # freeze anything that hit a boundary
        z = r(np.where(np.abs(z) > 1 - 1e-9, np.sign(z), z))

    # B0 @ z measured in full float64 (row 0 is the e_0 projection)
    return B0 @ z

_B = _A = _CHOP = None


def _init(B, a, sig_bits):
    global _B, _A, _CHOP
    _B, _A = B, a
    _CHOP = None if sig_bits == 52 else lpla.make_round(sig_bits)


def _rollout(seed_seq):
    return lower_bound_walk(_B, _A, _CHOP, seed_seq)


def run_lb(B, sig_bits, num_samples, workers, seed):
    """num_samples independent adversarial walks on B; returns (m, S) of Bz."""
    a = 2.0 ** (-sig_bits)
    seed_seqs = np.random.SeedSequence(seed).spawn(num_samples)
    if workers == 1:
        _init(B, a, sig_bits)
        out = [_rollout(s) for s in seed_seqs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init,
            initargs=(B, a, sig_bits),
        ) as ex:
            chunk = max(1, num_samples // (workers * 4))
            out = list(ex.map(_rollout, seed_seqs, chunksize=chunk))
    return np.stack(out, axis=1)


def _lb_directions(B, num_dirs, rng):
    """e_0 (adversarial axis) + top-10 left SVD dirs + random unit vectors."""
    m = B.shape[0]
    e0 = np.zeros(m)
    e0[0] = 1.0
    U, _, _ = np.linalg.svd(B, full_matrices=False)
    k = min(10, U.shape[1])
    n_rand = max(1, num_dirs - 1 - k)
    rand = rng.standard_normal((n_rand, m))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    return np.vstack([e0[None, :], U[:, :k].T, rand])


def _save_cache(cache_path, stats, sig_bits_values, n_values):
    keys = list(stats[(sig_bits_values[0], n_values[0])].keys())
    np.savez(cache_path, sig_bits=np.array(sig_bits_values),
             n_values=np.array(n_values),
             **{k: np.array([[stats[(b, v)][k] for v in n_values]
                             for b in sig_bits_values])
                for k in keys})
    print(f"cached stats -> {cache_path}")


def _load_cache(cache_path):
    z = np.load(cache_path)
    sig_bits_values = [int(b) for b in z["sig_bits"]]
    n_values = [int(v) for v in z["n_values"]]
    stats = {}
    for i, b in enumerate(sig_bits_values):
        for j, v in enumerate(n_values):
            stats[(b, v)] = {k: float(z[k][i, j])
                             for k in z.files
                             if k not in ("sig_bits", "n_values")}
    return stats, sig_bits_values, n_values


def plot_n_sweep(stats, sig_bits_values, n_values, num_samples, t,
                 save_path):
    colors = plt.cm.viridis(np.linspace(0, 1, len(sig_bits_values)))
    fig, (ax_s, ax_e, ax_d) = plt.subplots(1, 3, figsize=(16, 4.2))
    fig.suptitle(f"Lower-bound matrix: subgaussianity vs n "
                 f"({num_samples} walks; open markers violate a < 1/8n)")

    def mk(sig_bits, n):  # marker fill: open when outside theorem regime
        return "none" if not _valid_for_adversarial(n, sig_bits) else None

    for sig_bits, col in zip(sig_bits_values, colors):
        s = [stats[(sig_bits, n)] for n in n_values]
        for n, x in zip(n_values, s):
            ax_s.errorbar([n], [x["sig_max"]], yerr=[[x["sig_max_err"]],
                          [x["sig_max_err"]]], marker="o", markersize=4,
                          color=col, markerfacecolor=mk(sig_bits, n))
        ax_s.plot(n_values, [x["sig_max"] for x in s], color=col, lw=0.8)
        ax_s.plot([], [], "o", color=col, label=_lab(sig_bits))
        ax_s.plot(n_values, [x["psi2_max"] for x in s], ":", color=col,
                  marker="x", markersize=4)
    ax_s.axhline(1.0, color="gray", linestyle="--", lw=0.8, label="σ = 1")
    s_lo = stats[(min(sig_bits_values), n_values[0])]["sig_max"]
    ax_s.plot(n_values, s_lo * (np.array(n_values) / n_values[0]) ** 2,
              "k:", lw=0.8, alpha=0.6, label="∝ n²")
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("n")
    ax_s.set_ylabel("σ̂ (worst direction)")
    ax_s.set_title("max σ̂ vs n (dotted: ψ₂ norm)")
    ax_s.legend(title="format", fontsize=7)

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
    ax_e.set_title("worst-direction exceedance")
    ax_e.legend(title="format", fontsize=7)

    SIG_FLOOR = 1e-2  # below this, directions are deterministic (d·Bz const)
    for sig_bits, col in zip(sig_bits_values, colors):
        s = [stats[(sig_bits, n)] for n in n_values]
        ax_d.fill_between(n_values,
                          [max(x["sig_p10"], SIG_FLOOR) for x in s],
                          [x["sig_p90"] for x in s], color=col, alpha=0.2)
        ax_d.plot(n_values, [x["sig_med"] for x in s], marker="s",
                  markersize=4, color=col, label=_lab(sig_bits))
    ax_d.axhline(SIG_FLOOR, color="lightgray", linestyle=":", lw=0.8)
    ax_d.axhline(1.0, color="gray", linestyle="--", lw=0.8)
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("n")
    ax_d.set_ylabel("σ̂ over directions")
    ax_d.set_title("median σ̂ vs n (band: 10–90%)")
    ax_d.legend(title="format", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")


def plot_bits_sweep(stats, sig_bits_values, n, num_samples, save_path):
    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(f"Lower-bound matrix: σ̂ vs mantissa bits at n={n} "
                 f"({num_samples} walks)")

    sig_max = [stats[(b, n)]["sig_max"] for b in sig_bits_values]
    psi2_max = [stats[(b, n)]["psi2_max"] for b in sig_bits_values]
    errs = [stats[(b, n)]["sig_max_err"] for b in sig_bits_values]
    ax_s.errorbar(sig_bits_values, sig_max, yerr=errs, marker="o",
                  markersize=4, label="moment est (worst dir)")
    ax_s.plot(sig_bits_values, psi2_max, ":", marker="x", markersize=4,
              label="ψ₂ norm (worst dir)")
    ax_s.axhline(1.0, color="gray", linestyle="--", lw=0.8, label="σ = 1")
    # n^2 * 2^-b guide fit through the two lowest bit widths
    b0 = np.array(sig_bits_values[:2], dtype=float)
    s0 = np.array(sig_max[:2])
    c = float(np.mean(s0 * 2.0 ** b0))
    xs = np.linspace(min(sig_bits_values), max(sig_bits_values), 200)
    ax_s.plot(xs, c * 2.0 ** (-xs), "r--", lw=0.9, alpha=0.7,
              label=f"{c:.0f}·2⁻ᵇ  (∝ a·n²)")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("mantissa bits (sig_bits)")
    ax_s.set_ylabel("σ̂ (worst direction)")
    ax_s.set_title("σ̂ vs bits")
    ax_s.legend(fontsize=8)

    med = [stats[(b, n)]["sig_med"] for b in sig_bits_values]
    p10 = [stats[(b, n)]["sig_p10"] for b in sig_bits_values]
    p90 = [stats[(b, n)]["sig_p90"] for b in sig_bits_values]
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
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Subgaussianity figures for the lower-bound matrix")
    parser.add_argument("--n", type=int, nargs="+",
                        default=[50, 100, 200, 400, 800, 1600])
    parser.add_argument("--sig-bits", type=int, nargs="+",
                        default=[2, 3, 7, 10, 23, 52])
    parser.add_argument("--n-fixed", type=int, default=1000,
                        help="fixed n for the bits sweep (default 1000)")
    parser.add_argument("--sig-bits-sweep", type=int, nargs="+",
                        default=list(range(2, 21)) + [30, 40, 52])
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--dirs", type=int, default=64)
    parser.add_argument("--t", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_cache = figure_path("lb_subgauss", "lb_n_subgauss_cache.npz")
    b_cache = figure_path("lb_subgauss", "lb_bits_subgauss_cache.npz")

    if args.plot_only:
        stats_n, sb_n, n_vals = _load_cache(n_cache)
        stats_b, sb_b, n_b = _load_cache(b_cache)
        plot_n_sweep(stats_n, sb_n, n_vals, args.num_samples, args.t,
                     figure_path("lb_subgauss", "lb_n_subgauss.png"))
        plot_bits_sweep(stats_b, sb_b, n_b[0], args.num_samples,
                        figure_path("lb_subgauss", "lb_bits_subgauss.png"))
        return

    # figure 1: n sweep at each fp format
    stats_n = {}
    for sig_bits in tqdm(args.sig_bits, desc="n-sweep sig_bits"):
        for n in args.n:
            B = lower_bound_matrix(n)
            dirs = _lb_directions(B, args.dirs, rng)
            bz = run_lb(B, sig_bits, args.num_samples, args.workers,
                        args.seed)
            stats_n[(sig_bits, n)] = collect(dirs @ bz, args.t)
    _save_cache(n_cache, stats_n, args.sig_bits, args.n)
    plot_n_sweep(stats_n, args.sig_bits, args.n, args.num_samples, args.t,
                 figure_path("lb_subgauss", "lb_n_subgauss.png"))

    # figure 2: bits sweep at fixed n
    n = args.n_fixed
    B = lower_bound_matrix(n)
    dirs = _lb_directions(B, args.dirs, rng)
    stats_b = {}
    for sig_bits in tqdm(args.sig_bits_sweep, desc="bits-sweep"):
        bz = run_lb(B, sig_bits, args.num_samples, args.workers, args.seed)
        stats_b[(sig_bits, n)] = collect(dirs @ bz, args.t)
    _save_cache(b_cache, stats_b, args.sig_bits_sweep, [n])
    plot_bits_sweep(stats_b, args.sig_bits_sweep, n, args.num_samples,
                    figure_path("lb_subgauss", "lb_bits_subgauss.png"))


if __name__ == "__main__":
    main()
