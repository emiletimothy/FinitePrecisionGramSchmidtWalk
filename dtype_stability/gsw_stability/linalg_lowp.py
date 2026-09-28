"""Precision-sensitive linear algebra, with every arithmetic op rounded to the
dtype of its inputs.

numpy's float16 ufuncs (+, -, *, /, sqrt) round each result to float16, but its
reductions (np.sum, np.dot, @, np.linalg) accumulate float16 in float32 or call
LAPACK. Nothing here uses them: every reduction goes through `tree_sum`, which
is built only from elementwise adds. The same routines are used for every
dtype, so runs at different precisions differ only in rounding.
"""
import numpy as np


class BreakdownError(ArithmeticError):
    """A factorization or downdate lost positive definiteness."""


def _zero(dt):
    return np.dtype(dt).type(0)


def tree_sum(x, axis=0):
    """Pairwise sum along `axis`: repeatedly add the first half onto the second."""
    x = np.moveaxis(np.asarray(x), axis, 0)
    if x.shape[0] == 0:
        return np.zeros(x.shape[1:], dtype=x.dtype)
    while x.shape[0] > 1:
        k = x.shape[0]
        h = k // 2
        s = x[:h] + x[h:2 * h]
        if k % 2:
            s = np.concatenate([s, x[2 * h:]], axis=0)
        x = s
    return x[0]


def dot(x, y):
    return tree_sum(x * y)


def matvec(A, x):
    """A @ x."""
    return tree_sum(A * x, axis=1)


def rmatvec(A, y):
    """A.T @ y."""
    return tree_sum(A * y[:, None], axis=0)


def gram(A):
    """A.T @ A, one column at a time (exactly symmetric)."""
    k = A.shape[1]
    G = np.empty((k, k), dtype=A.dtype)
    for j in range(k):
        G[:, j] = rmatvec(A, A[:, j])
    return G


def nrm2(x):
    """Euclidean norm, scaled by max|x| (as LAPACK does) to avoid under/overflow."""
    if x.size == 0:
        return _zero(x.dtype)
    scale = np.max(np.abs(x))
    if scale == 0 or not np.isfinite(scale):
        return scale
    y = x / scale
    return scale * np.sqrt(dot(y, y))


# ---------------------------------------------------------------- Householder QR

def householder_qr(A):
    """Householder QR (LAPACK geqr2/larfg conventions) in A's dtype.

    Returns (QR, tau): R in the upper triangle, reflector tails v[1:] below it
    (v[0] = 1), so that H_j = I - tau_j v v^T and Q = H_0 H_1 ... H_{k-1}.
    """
    A = np.array(A, copy=True)
    m, k = A.shape
    dt = A.dtype
    tau = np.zeros(k, dtype=dt)
    for j in range(k):
        alpha = A[j, j]
        x = A[j + 1:, j]
        xnorm = nrm2(x)
        if xnorm == 0:
            continue
        beta = -np.copysign(nrm2(np.array([alpha, xnorm], dtype=dt)), alpha)
        tau[j] = (beta - alpha) / beta
        A[j + 1:, j] = x / (alpha - beta)
        A[j, j] = beta
        if j + 1 < k:
            v = np.concatenate([np.ones(1, dtype=dt), A[j + 1:, j]])
            sub = A[j:, j + 1:]
            w = tau[j] * rmatvec(sub, v)
            sub -= v[:, None] * w
    return A, tau


def apply_qt(QR, tau, b):
    """Q^T b for the reflectors stored by `householder_qr`."""
    b = np.array(b, copy=True)
    one = np.ones(1, dtype=b.dtype)
    for j in range(QR.shape[1]):
        if tau[j] == 0:
            continue
        v = np.concatenate([one, QR[j + 1:, j]])
        b[j:] -= (tau[j] * dot(v, b[j:])) * v
    return b


def solve_upper(R, y):
    """Solve R x = y (column-oriented back substitution); y may be 1-D or 2-D."""
    x = np.array(y, copy=True)
    for j in range(R.shape[0] - 1, -1, -1):
        x[j] = x[j] / R[j, j]
        if j:
            x[:j] -= np.multiply.outer(R[:j, j], x[j])
    return x


def solve_upper_t(U, y):
    """Solve U^T x = y for upper-triangular U (forward substitution)."""
    x = np.array(y, copy=True)
    n = U.shape[0]
    for j in range(n):
        x[j] = x[j] / U[j, j]
        if j + 1 < n:
            x[j + 1:] -= np.multiply.outer(U[j, j + 1:], x[j])
    return x


def qr_lstsq(A, b):
    """argmin_x ||A x - b||_2 via Householder QR and back substitution."""
    k = A.shape[1]
    if k == 0:
        return np.zeros(0, dtype=A.dtype)
    QR, tau = householder_qr(A)
    c = apply_qt(QR, tau, b)[:k]
    return solve_upper(QR[:k, :k], c)


# ---------------------------------------------------------------- Cholesky

def cholesky(M):
    """Upper Cholesky factor U with U^T U = M (right-looking)."""
    A = np.array(M, copy=True)
    n = A.shape[0]
    U = np.zeros_like(A)
    for j in range(n):
        d = A[j, j]
        if not d > 0:
            raise BreakdownError(f"cholesky: pivot {j} is {d}")
        ujj = np.sqrt(d)
        U[j, j] = ujj
        if j + 1 < n:
            row = A[j, j + 1:] / ujj
            U[j, j + 1:] = row
            A[j + 1:, j + 1:] -= np.multiply.outer(row, row)
    return U


def chol_solve(U, b):
    """Solve U^T U x = b."""
    return solve_upper(U, solve_upper_t(U, b))


def chol_inverse(M):
    """M^{-1} via Cholesky and n triangular solves."""
    U = cholesky(M)
    return chol_solve(U, np.eye(M.shape[0], dtype=M.dtype))


def _givens_bounds(dt):
    # Julia's givensAlgorithm (a port of LAPACK dlartg) rescales f, g by powers
    # of two when max(|f|, |g|) leaves [safmn2, safmx2].
    fi = np.finfo(dt)
    e = int(np.trunc(np.log2(float(fi.tiny) / float(fi.eps)) / 2))
    return dt.type(2.0 ** e), dt.type(2.0 ** -e)


def givens(f, g):
    """(c, s, r) with [c s; -s c] [f; g] = [r; 0], as Julia's givensAlgorithm."""
    dt = np.asarray(f).dtype
    if g == 0:
        return dt.type(1), dt.type(0), f
    if f == 0:
        return dt.type(0), dt.type(1), g
    scale = max(abs(f), abs(g))
    safmn2, safmx2 = _givens_bounds(dt)
    if scale >= safmx2 or scale <= safmn2:
        sc = dt.type(2.0 ** -np.frexp(float(scale))[1])  # exact power-of-two rescale
        f1, g1 = f * sc, g * sc
        r = np.sqrt(f1 * f1 + g1 * g1)
        c, s, r = f1 / r, g1 / r, r / sc
    else:
        r = np.sqrt(f * f + g * g)
        c, s = f / r, g / r
    if abs(f) > abs(g) and c < 0:
        c, s, r = -c, -s, -r
    return c, s, r


def chol_rank1_update(U, x):
    """In place: U^T U <- U^T U + x x^T (Julia LinearAlgebra.lowrankupdate!, uplo='U')."""
    x = np.array(x, copy=True)
    n = x.shape[0]
    for i in range(n):
        c, s, r = givens(U[i, i], x[i])
        U[i, i] = r
        if i + 1 < n:
            Uij = U[i, i + 1:].copy()
            xj = x[i + 1:].copy()
            U[i, i + 1:] = c * Uij + s * xj
            x[i + 1:] = -s * Uij + c * xj
    return U


def chol_rank1_downdate(U, x):
    """In place: U^T U <- U^T U - x x^T (Julia LinearAlgebra.lowrankdowndate!, uplo='U').

    Julia throws PosDefException when s^2 > 1; s^2 == 1 would divide by c = 0,
    so both raise BreakdownError here.
    """
    x = np.array(x, copy=True)
    n = x.shape[0]
    one = x.dtype.type(1)
    for i in range(n):
        Uii = U[i, i]
        s = x[i] / Uii
        s2 = s * s
        if not s2 < one:
            raise BreakdownError(f"downdate: s^2 = {s2} at row {i}")
        c = np.sqrt(one - s2)
        U[i, i] = c * Uii
        if i + 1 < n:
            xj = x[i + 1:].copy()
            Uij = (U[i, i + 1:] - s * xj) / c
            U[i, i + 1:] = Uij
            x[i + 1:] = -s * Uij + c * xj
    return U


# ---------------------------------------------------------------- inverse updates

def sm_remove(C, pos, q, qii):
    """Inverse of Q with row/column `pos` deleted, given C = Q^{-1}.

    Low-Rank Thinning (Carrell et al.), Alg. GS-Halve-Cubic:
    D = C without row/col pos, q = Q[kept, pos], and
    C_new = D - D q q^T D / (Q_pos,pos + q^T D q).
    """
    keep = np.ones(C.shape[0], dtype=bool)
    keep[pos] = False
    D = C[np.ix_(keep, keep)]
    Dq = matvec(D, q)
    qD = rmatvec(D, q)
    denom = qii + dot(q, Dq)
    return D - np.multiply.outer(Dq, qD) / denom
