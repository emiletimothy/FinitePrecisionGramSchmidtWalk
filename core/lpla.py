"""Low-precision linear algebra.

A hand-rolled, shape-adaptive least-squares solver (Householder QR / LQ) in which
every elementary operation runs at reduced precision, so the *solver itself* is low
precision — unlike np.linalg.lstsq (whole solve in float64) or pychop's operation
layer (rounds only a reduction's inputs/outputs). This is aimed at studying how
mantissa rounding affects the Gram-Schmidt Walk, not at bit-exact hardware emulation:

  * Only the *mantissa* is modeled (round to `sig_bits` fraction bits). The exponent
    range is unbounded, so overflow/underflow never occur — a deliberate simplification.
  * Products are rounded, but reductions (dot / matvec / norm) accumulate in float64
    and round once. Every op is thus reduced-precision, but the exact accumulation
    order — a second-order detail — is left to BLAS, which keeps it fast.

`r` throughout is a rounding callable that maps an ndarray to `sig_bits` mantissa bits.
At full precision (r = identity) every routine reproduces np.linalg.lstsq for full-rank
inputs; see the __main__ block for the correctness check.
"""

import numpy as np


def _mantissa_rounder(sig_bits, kind):
    """Significand rounder to `sig_bits` fraction bits; `kind` sets the direction
    (np.round = nearest-even, np.ceil = toward +inf, np.floor = toward -inf)."""
    step = 2.0 ** (sig_bits + 1)   # significand quantization (m in [0.5,1) -> sig_bits frac bits)

    def rnd(x):
        m, e = np.frexp(np.asarray(x, dtype=np.float64))   # x = m * 2**e, |m| in [0.5,1)
        return np.ldexp(kind(m * step) / step, e)          # round significand, rescale

    rnd.sig_bits = sig_bits   # lets consumers scale tolerances to the format ulp
    return rnd


def make_round(sig_bits):
    """Return a fast round-to-nearest-even *mantissa* rounder (`sig_bits` fraction bits).

    Rounds each value's significand to `sig_bits` fraction bits via frexp/ldexp — a
    handful of vectorized C ops — with an unbounded exponent, so overflow never happens.
    Only mantissa precision is modeled; the exponent range (and thus overflow/subnormal
    behavior) is deliberately not, since this is for studying the effect of rounding, not
    bit-exact hardware emulation. For in-range values this matches the rounding of an IEEE
    format with `sig_bits` fraction bits.

    The returned callable also carries `.up` and `.down`: directed variants that round
    the significand toward +inf and -inf, for interval endpoints that must not round
    outward. At sig_bits=52 all three act as the identity on float64 inputs.
    """
    rnd = _mantissa_rounder(sig_bits, np.round)
    rnd.up = _mantissa_rounder(sig_bits, np.ceil)
    rnd.down = _mantissa_rounder(sig_bits, np.floor)
    return rnd


def _dot(x, y, r):
    """Reduced-precision inner product: round the products, sum in fp64, round once."""
    return r(np.sum(r(x * y)))


def _norm(x, r):
    """Reduced-precision Euclidean norm."""
    return r(np.sqrt(r(np.sum(r(x * x)))))


def _rowdot(v, A, r):
    """vᵀ A for every column at once (1-D array of length A.shape[1]).

    Rounds the per-row products then sums over rows in fp64, rounding once.
    """
    return r(np.sum(r(v[:, None] * A), axis=0))


def householder_qr(A, r, round_inputs=True):
    """Householder QR with per-op rounding.

    Returns (V, betas, R) where the reflectors are H_j = I - betas[j] * outer(v_j, v_j)
    with v_j stored in V[:, j] (implicit-zero above row j), and R is upper triangular.
    A is (m, k) with m >= k assumed by callers via lstsq's shape dispatch.
    round_inputs=False keeps the stored copy of A exact; computed ops still round.
    """
    m, k = A.shape
    R = r(A.copy()) if round_inputs else A.copy()
    V = np.zeros((m, k))
    betas = np.zeros(k)

    for j in range(k):
        x = R[j:, j]
        alpha = _norm(x, r)
        if alpha == 0.0:
            continue
        # Choose sign to avoid cancellation: alpha carries -sign(x[0]).
        if x[0] < 0:
            alpha = -alpha
        v = x.copy()
        v[0] = r(v[0] + alpha)          # v = x + sign*alpha*e_1
        vnorm2 = _dot(v, v, r)
        if vnorm2 == 0.0:
            continue
        beta = r(2.0 / vnorm2)
        # Apply H to the trailing block R[j:, j:]
        block = R[j:, j:]
        w = _rowdot(v, block, r)                       # vᵀ block
        R[j:, j:] = r(block - r(beta * r(np.outer(v, w))))
        V[j:, j] = v
        betas[j] = beta
    return V, betas, R


def apply_QT(V, betas, c, r, round_inputs=True):
    """Apply Qᵀ = H_{k-1} ... H_0 to vector c (forward reflector order)."""
    c = r(c.copy()) if round_inputs else c.copy()
    k = V.shape[1]
    for j in range(k):
        if betas[j] == 0.0:
            continue
        v = V[j:, j]
        s = _dot(v, c[j:], r)                          # vᵀ c
        c[j:] = r(c[j:] - r(r(betas[j] * s) * v))
    return c


def apply_Q(V, betas, y, r, round_inputs=True):
    """Apply Q = H_0 ... H_{k-1} to vector y (reverse reflector order)."""
    y = r(y.copy()) if round_inputs else y.copy()
    k = V.shape[1]
    for j in range(k - 1, -1, -1):
        if betas[j] == 0.0:
            continue
        v = V[j:, j]
        s = _dot(v, y[j:], r)
        y[j:] = r(y[j:] - r(r(betas[j] * s) * v))
    return y


def solve_upper(R, b, r, round_inputs=True):
    """Back-substitution for upper-triangular R x = b, per-op rounded."""
    k = R.shape[1]
    x = np.zeros(k)
    for i in range(k - 1, -1, -1):
        s = r(b[i]) if round_inputs else float(b[i])
        if i + 1 < k:
            s = r(s - _dot(R[i, i + 1:], x[i + 1:], r))
        x[i] = r(s / R[i, i])
    return x


def solve_lower(L, b, r, round_inputs=True):
    """Forward-substitution for lower-triangular L x = b, per-op rounded."""
    k = L.shape[0]
    x = np.zeros(k)
    for i in range(k):
        s = r(b[i]) if round_inputs else float(b[i])
        if i > 0:
            s = r(s - _dot(L[i, :i], x[:i], r))
        x[i] = r(s / L[i, i])
    return x


def lstsq(A, c, r, round_inputs=True):
    """Solve min_v ||A v - c|| with every operation rounded by `r`.

    Shape-adaptive to match np.linalg.lstsq's minimum-norm least-squares solution for
    full-rank A:
      tall/square (m >= k): QR, then R v = Qᵀ c.
      wide        (m <  k): min-norm via LQ (QR of Aᵀ), R₁ᵀ w = c, v = Q₁ w.

    round_inputs=False keeps A and c exact: only computed quantities are rounded.
    This isolates arithmetic error from input-quantization error.
    """
    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=float)
    if round_inputs:
        A = r(A)
        c = r(c)
    m, k = A.shape

    with np.errstate(over="ignore", invalid="ignore"):
        if m >= k:
            V, betas, R = householder_qr(A, r, round_inputs)
            y = apply_QT(V, betas, c, r, round_inputs)
            v = solve_upper(R[:k, :k], y[:k], r, round_inputs)
        else:
            # QR of Aᵀ = Q₁ R₁ (R₁ is m×m upper). A = R₁ᵀ Q₁ᵀ.
            V, betas, R = householder_qr(A.T.copy(), r, round_inputs)
            w = solve_lower(R[:m, :m].T, c, r, round_inputs)  # R₁ᵀ w = c (lower-tri)
            y = np.zeros(k)
            y[:m] = w
            v = apply_Q(V, betas, y, r, round_inputs)

    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    # Correctness: at full precision the hand-rolled solver must match np.linalg.lstsq
    # for full-rank tall, wide, and square systems.
    ident = lambda x: np.asarray(x, dtype=float)
    rng = np.random.default_rng(0)
    for (m, k) in [(500, 99), (30, 300), (100, 100), (50, 50)]:
        A = rng.standard_normal((m, k))
        c = rng.standard_normal(m)
        mine = lstsq(A, c, ident)
        ref, *_ = np.linalg.lstsq(A, c, rcond=None)
        err = np.linalg.norm(mine - ref) / max(np.linalg.norm(ref), 1e-300)
        res_mine = np.linalg.norm(A @ mine - c)
        res_ref = np.linalg.norm(A @ ref - c)
        print(f"({m:3d}x{k:3d})  rel-err vs numpy = {err:.2e}   "
              f"residual mine={res_mine:.4e} numpy={res_ref:.4e}   "
              f"||v|| mine={np.linalg.norm(mine):.4e} numpy={np.linalg.norm(ref):.4e}")
        assert err < 1e-9, f"mismatch at ({m},{k}): {err}"
    print("OK: lpla.lstsq matches np.linalg.lstsq at full precision.")
