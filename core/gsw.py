import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from . import lpla


@dataclass
class WalkResult:
    assignment: np.ndarray
    # weighted sum of input columns by final assignment: B0 @ assignment,
    # measured in full float64 against the *unrounded* input matrix
    Bz: np.ndarray
    # B_hat @ assignment: the same assignment scored against the matrix the
    # walk actually saw (round(B0) when quantize_input, else B0). The
    # difference Bz - Bz_hat = (B0 - B_hat) z isolates input quantization.
    Bz_hat: np.ndarray
    # number of random steps taken
    steps: int = 0
    # early-termination reason; None when the walk froze every coordinate
    failed: str | None = None
    # Each entry is (z_before_step, u, delta_t). Only populated when
    # gram_schmidt_walk is called with record_trajectory=True.
    trajectory: list[tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)


def gram_schmidt_walk(
    B: np.ndarray,
    chop: Callable | None = None,
    noise: Callable[[int], np.ndarray] | None = None,
    record_trajectory: bool = False,
    quantize_input: bool = True,
    snap_c: float = 10.0,
    max_steps: int | None = None,
) -> WalkResult:
    """
    Gram-Schmidt Walk (Algorithm 1).

    B:     (m, n) matrix with unit-norm columns
    chop:  optional callable that rounds an array to a reduced-precision format.
           When given, the least-squares direction is computed by lpla.lstsq, a
           hand-rolled Householder QR/LQ solver in which every arithmetic operation
           is rounded to the target format (not just the solve's inputs/output).
    noise: optional callable(n) -> ndarray; its output is added to z after each
           step, and z is clamped back to [-1, 1]^n before the next iteration.
           Example: lambda n: np.random.normal(0, 0.01, n)
    quantize_input: round the matrix the direction solves see up front
           (B_solve = chop(B0)). When False the solves read the exact input while
           all computed arithmetic is still rounded — isolating arithmetic error
           from representation error. Bz_hat is always scored against B_hat =
           chop(B0), so Bz - Bz_hat isolates pure input quantization either way.
    snap_c: snap tolerance as a multiple of the format ulp (10 * eps at fp64);
           coordinates landing within snap_tol = snap_c * ulp of +-1 freeze.
    max_steps: early termination after this many steps (failed="step_limit").
    Returns: WalkResult with assignment vector in {-1, +1}^n and run statistics
    """
    r = chop if chop is not None else lambda x: x
    B0 = np.asarray(B, dtype=float)
    B_hat = r(B0.copy())
    B = B_hat if quantize_input else B0
    _, n = B.shape
    # Precision-scaled tolerances (cf. walks._walk in dtype_stability): the
    # activity test is exact (|z| < 1); coordinates within snap_tol of the
    # boundary are snapped/frozen. snap_tol is capped well below the coarse
    # rounding grid — at 2-bit precision 10*ulp = 2.5 would freeze the whole
    # cube after a single step.
    ulp = 2.0 ** (-getattr(chop, "sig_bits", 52))
    snap_tol = min(snap_c * ulp, 1e-9)
    z = np.zeros(n)
    p = np.random.randint(n)
    steps = 0
    failed = None
    trajectory = []

    # Activity is exact: coordinates stay live until they sit exactly on +-1.
    while np.any(np.abs(z) < 1.0):
        if max_steps is not None and steps >= max_steps:
            failed = "step_limit"
            break

        active = np.flatnonzero(np.abs(z) < 1.0)
        if p not in active:
            p = active[np.random.randint(len(active))]

        # Step direction: argmin_u ||Bu||^2  s.t.  u[p]=1, u[i]=0 for i not in active
        # Substituting u[p]=1, minimise ||B_free @ v + B_p||^2 over free variables v.
        free = active[active != p]
        u = np.zeros(n)
        u[p] = 1.0
        if len(free) > 0:
            if chop is not None:
                # Every op in the solve runs at the target precision, but input
                # rounding is managed here via quantize_input (round_inputs=False
                # so lstsq does not re-round A — it is already rounded, or exact
                # on request). Overflow at low precision yields inf/nan, which is
                # zeroed so the step falls back to the pivot direction.
                v = lpla.lstsq(B[:, free], -B[:, p], chop, round_inputs=False)
            else:
                v, _, _, _ = np.linalg.lstsq(B[:, free], -B[:, p], rcond=None)
            u[free] = v
        u[~np.isfinite(u)] = 0.0   # nonfinite coefficients -> zero
        u = r(u)

        # Feasible step interval Delta = {delta : z + delta*u in [-1,1]^n}.
        # The pivot (u_p = 1) participates in both caps: 1 - z_p upward and
        # 1 + z_p downward.
        nz = np.flatnonzero(np.abs(u) > 0.0)   # u == 0 yields +-inf bounds anyway
        _z = z[nz]
        _u = u[nz]
        r1 = r((-1.0 - _z) / _u)
        r2 = r(( 1.0 - _z) / _u)
        lo = r(np.minimum(r1, r2))
        hi = r(np.maximum(r1, r2))
        delta_min = float(r(np.array(np.max(lo))))
        delta_max = float(r(np.array(np.min(hi))))

        d_plus  = float(r(np.array(abs(delta_max))))   # |max Delta|
        d_minus = float(r(np.array(abs(delta_min))))   # |min Delta|
        total = float(r(np.array(d_plus + d_minus)))
        if not np.isfinite(total) or total == 0.0:
            failed = "degenerate_interval"
            break

        prob = float(r(np.array(d_minus / total)))
        if not np.isfinite(prob):
            failed = "nonfinite_prob"
            break
        # Martingale-preserving random step: E[delta_t] = 0
        delta_t = d_plus if np.random.random() < prob else -d_minus
        delta_t = float(r(np.array(delta_t)))
        if record_trajectory:
            trajectory.append((z.copy(), u.copy(), delta_t))

        z = r(z + r(delta_t * u))
        # Coordinates attaining the step cap land exactly on their bound
        # (forced snap), instead of relying on the arithmetic to hit +-1.
        if delta_t > 0:
            hits = nz[hi == delta_max]
            z[hits] = np.sign(u[hits])
        else:
            hits = nz[lo == delta_min]
            z[hits] = -np.sign(u[hits])

        if noise is not None:
            unfrozen = np.abs(z) < 1.0
            _noise = r(noise(unfrozen.sum()))
            z[unfrozen] = r(z[unfrozen] + _noise)

        # Clamp overshoots and snap near-boundary coordinates, then freeze.
        az = np.abs(z)
        snap = (az >= 1.0 - snap_tol) & (az != 1.0)
        z[snap] = np.sign(z[snap])
        z = r(z)
        steps += 1

    z = np.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)
    assignment = np.sign(z).astype(int)
    # final discrepancies are measured in full float64, independent of the
    # rounded walk state; Bz_hat scores the same assignment against B_hat.
    return WalkResult(
        assignment=assignment,
        Bz=B0 @ assignment,
        Bz_hat=B_hat @ assignment,
        steps=steps,
        failed=failed,
        trajectory=trajectory if record_trajectory else [],
    )
