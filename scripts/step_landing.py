"""Per-iteration "step landing" error: rounding as an effective additive epsilon.

Walk the EXACT fp64 Gram-Schmidt Walk trajectory and, at every iteration, also compute
the low-precision fixed-point step on the identical state. The additive difference
    eps = u_lowprec - u_ideal      (over the free coordinates)
is the effective per-step perturbation that rounding injects — the analogue of the
additive noise the sweeps inject into z (noise_mean/noise_std mode). This is a fresh,
per-time-step comparison: the trajectory itself never accumulates rounding error (we
always advance with the ideal step).

For each step we record the direction epsilon and the z-space epsilon (delta_t * eps),
the latter being in the same place/units as the injected noise (z += noise), so its
spread is directly comparable to a noise_std.

The walk loop is an instrumented copy of gsw.gram_schmidt_walk (gsw.py) — we need both
solves on the same state and must advance along the ideal path, which the library call
doesn't expose.
"""

import argparse
import os
os.environ.setdefault("chop_backend", "numpy")

import matplotlib
matplotlib.use("Agg")  # non-interactive: save figures, never pop up a window
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import _bootstrap  # noqa: F401  # repo root on sys.path
from core import lpla, matrices
from core.paths import figure_path


def _one_walk(B: np.ndarray, chop):
    """One ideal (fp64) walk; at each step also solve in low precision and record eps.

    Yields per step: (t, k_free, eps_dir, eps_z) where eps_dir = u_lp - u_id over the
    free coords and eps_z = delta_t * eps_dir (z-space perturbation).
    """
    _, n = B.shape
    z = np.zeros(n)
    p = np.random.randint(n)
    t = 0
    records = []

    while True:
        active = np.where(np.abs(z) < 1 - 1e-9)[0]
        if len(active) == 0:
            break
        if p not in active:
            p = active[np.random.randint(len(active))]

        free = active[active != p]
        u = np.zeros(n)
        u[p] = 1.0
        eps_dir = None
        if len(free) > 0:
            b_p = -B[:, p]
            v_id, _, _, _ = np.linalg.lstsq(B[:, free], b_p, rcond=None)  # ideal
            v_lp = lpla.lstsq(B[:, free], b_p, chop)                      # low precision
            eps_dir = v_lp - v_id
            u[free] = v_id  # advance along the IDEAL step (no accumulation)

        # Feasible step interval + martingale step, all from the ideal direction u
        nz = np.abs(u) > 1e-15
        r1 = (-1 - z[nz]) / u[nz]
        r2 = ( 1 - z[nz]) / u[nz]
        delta_min = np.max(np.minimum(r1, r2))
        delta_max = np.min(np.maximum(r1, r2))
        d_plus, d_minus = abs(delta_max), abs(delta_min)
        total = d_plus + d_minus
        if total < 1e-15:
            break
        delta_t = d_plus if np.random.random() < d_minus / total else -d_minus

        if eps_dir is not None:
            records.append((t, len(free), eps_dir, delta_t * eps_dir))

        z += delta_t * u
        z = np.where(np.abs(z) > 1 - 1e-9, np.sign(z), z)
        t += 1

    return records


def _quart(rows_by_t, key):
    """Given {t: list of scalars}, return sorted t and per-t (mean, q1, median, q3)."""
    ts = sorted(rows_by_t)
    mean = np.array([np.mean(rows_by_t[t]) for t in ts])
    q1   = np.array([np.percentile(rows_by_t[t], 25) for t in ts])
    med  = np.array([np.percentile(rows_by_t[t], 50) for t in ts])
    q3   = np.array([np.percentile(rows_by_t[t], 75) for t in ts])
    return np.array(ts), mean, q1, med, q3


def step_landing(matrix: str, m: int, n: int, sig_bits: int,
                 samples: int = 100, *, seed=None, save_path=None) -> None:
    np.random.seed(np.random.SeedSequence(seed).generate_state(4))
    B = matrices.matrix_family(matrix, m=m, n_max=n, seed=seed)(n)
    m = B.shape[0]
    chop = lpla.make_round(sig_bits)

    # collections keyed by free-set size k (pins the k≈m spike regardless of n)
    norm_dir, rms_dir, rms_z = {}, {}, {}
    pooled_dir, pooled_z = [], []  # per-coordinate eps values, all steps/samples

    seeds = np.random.SeedSequence(seed).spawn(samples)
    for ss in tqdm(seeds, desc=f"walks {matrix} m={m} n={n} s={sig_bits}"):
        np.random.seed(ss.generate_state(4))
        for t, k, eps_dir, eps_z in _one_walk(B, chop):
            norm_dir.setdefault(k, []).append(np.linalg.norm(eps_dir))
            rms_dir.setdefault(k, []).append(np.linalg.norm(eps_dir) / np.sqrt(k))
            rms_z.setdefault(k, []).append(np.linalg.norm(eps_z) / np.sqrt(k))
            pooled_dir.append(eps_dir)
            pooled_z.append(eps_z)

    pooled_dir = np.concatenate(pooled_dir)
    pooled_z = np.concatenate(pooled_z)

    def _summary(name, x):
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        # normal-consistent robust sigma from the IQR (IQR = 1.349 sigma for a normal);
        # immune to the rare k≈m ill-conditioning blowups that dominate the raw std.
        sig_iqr = (q3 - q1) / 1.349
        print(f"  {name:22s} mean={x.mean():+.3e}  sigma_IQR={sig_iqr:.3e}  "
              f"Q1={q1:+.3e}  med={med:+.3e}  Q3={q3:+.3e}  (raw std={x.std():.2e})")
        return sig_iqr

    u_round = 2.0 ** (-(sig_bits + 1))
    print(f"\nStep-landing epsilon  ({matrix} m={m}, n={n}, sig_bits={sig_bits}, samples={samples})")
    print(f"unit roundoff u = 2^-(sig_bits+1) = {u_round:.3e}")
    print("pooled per-coordinate additive epsilon (rounding step - ideal step):")
    _summary("direction eps", pooled_dir)
    sig_z = _summary("z-space eps (delta*eps)", pooled_z)
    print(f"  -> effective additive-noise sigma (z-space, IQR-based) = {sig_z:.3e}  "
          f"(compare to sweep noise_std, e.g. 2^-8={2**-8:.3e}, 2^-10={2**-10:.3e})")

    # ---- plot: mean + IQR vs free-set size k (x-axis descending = walk progression) ----
    k_a, mean_a, q1_a, med_a, q3_a = _quart(norm_dir, None)
    k_b, _, q1_b, med_b, q3_b = _quart(rms_dir, None)
    k_z, _, _, med_z, _ = _quart(rms_z, None)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"GSW step-landing error vs free-set size  —  {matrix} "
                 f"m={m}, n={n}, sig_bits={sig_bits}, {samples} samples")

    axA.fill_between(k_a, q1_a, q3_a, alpha=0.25, label="IQR (25–75)")
    axA.plot(k_a, med_a, lw=1.5, label="median")
    axA.plot(k_a, mean_a, "--", lw=1, label="mean")
    axA.axvline(m, color="tab:red", ls="--", lw=0.8, label=f"k = m = {m}")
    axA.set_xlabel("free-set size k  (walk runs right → left)")
    axA.set_ylabel(r"absolute direction error $\|u_{lp}-u_{id}\|_2$")
    axA.set_yscale("log")  # spike at k≈m dwarfs the rest on a linear scale
    axA.invert_xaxis()      # large k (walk start) on the left, so time reads left → right
    axA.legend()

    axB.fill_between(k_b, q1_b, q3_b, alpha=0.25, label="direction IQR")
    axB.plot(k_b, med_b, lw=1.5, label="direction median")
    axB.plot(k_z, med_z, lw=1.5, label="z-space median")
    axB.axhline(u_round, color="gray", ls=":", lw=0.8, label=f"unit roundoff 2^-{sig_bits+1}")
    axB.axvline(m, color="tab:red", ls="--", lw=0.8, label=f"k = m = {m}")
    axB.set_xlabel("free-set size k  (walk runs right → left)")
    axB.set_ylabel(r"per-coordinate epsilon RMS ($\|\epsilon\|_2/\sqrt{k}$)")
    axB.set_yscale("log")
    axB.invert_xaxis()
    axB.legend()

    fig.tight_layout()
    savepath = save_path or figure_path(
        "step_landing", f"step_landing_{matrix}_m{m}_n{n}_s{sig_bits}.png")
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"saved {savepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    matrices.add_matrix_arguments(parser, default="clustered")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sig-bits", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    step_landing(matrix=args.matrix, m=args.m, n=args.n,
                 sig_bits=args.sig_bits, samples=args.samples,
                 seed=args.seed, save_path=args.save)
