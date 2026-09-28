"""Distributional metrics, all computed in float64 from the final signs."""
import numpy as np
from scipy.special import logsumexp

MOMENTS = (2, 4, 6, 8)


def test_directions(B, k=20, seed=12345):
    """k fixed random unit vectors plus the top and bottom left singular vectors of B."""
    m = B.shape[0]
    V = np.random.default_rng(seed).standard_normal((m, k))
    V /= np.linalg.norm(V, axis=0)
    Ub = np.linalg.svd(B, full_matrices=False)[0]
    names = [f"rand{j}" for j in range(k)] + ["top_sv", "bottom_sv"]
    return np.column_stack([V, Ub[:, 0], Ub[:, -1]]), names


def psi2_orlicz(y):
    """Smallest K with mean(exp(y^2 / K^2)) <= 2, by bisection in log K."""
    y2 = np.square(np.asarray(y, dtype=float))
    if not np.any(y2 > 0):
        return 0.0
    logn2 = np.log(len(y2)) + np.log(2)

    def excess(K):
        return logsumexp(y2 / K**2) - logn2

    hi = np.sqrt(y2.max() / np.log(2))  # every term <= 2 here
    lo = hi * 1e-3
    while excess(lo) <= 0:
        lo *= 1e-3
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if excess(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return hi


def psi2_moment(y):
    """max_p (E|y|^p)^{1/p} / sqrt(p) over p in MOMENTS."""
    a = np.abs(np.asarray(y, dtype=float))
    return max(np.mean(a**p) ** (1 / p) / np.sqrt(p) for p in MOMENTS)


def summarize(Y):
    """Per-direction statistics for Y of shape (runs, directions)."""
    R = Y.shape[0]
    return {
        "orlicz": np.array([psi2_orlicz(Y[:, j]) for j in range(Y.shape[1])]),
        "moment": np.array([psi2_moment(Y[:, j]) for j in range(Y.shape[1])]),
        "mean": Y.mean(axis=0),
        "se": Y.std(axis=0, ddof=1) / np.sqrt(R) if R > 1 else np.full(Y.shape[1], np.nan),
    }


def ccdf(y):
    """(t, P(|Y| >= t)) at the sorted sample points."""
    t = np.sort(np.abs(y))[::-1]
    return t, np.arange(1, len(t) + 1) / len(t)


def first_divergence(hit_a, piv_a, hit_b, piv_b):
    """First step at which two paths differ in pivot or in which coordinates hit +-1.

    hit_* is the step at which each coordinate first reached +-1 (-1 if never);
    piv_* is the pivot at each step. Returns -1 if the paths agree.
    """
    big = np.iinfo(np.int64).max
    hit_a, hit_b = np.asarray(hit_a, np.int64), np.asarray(hit_b, np.int64)
    ha = np.where(hit_a < 0, big, hit_a)
    hb = np.where(hit_b < 0, big, hit_b)
    cands = []
    diff = ha != hb
    if diff.any():
        cands.append(np.minimum(ha[diff], hb[diff]).min())
    L = min(len(piv_a), len(piv_b))
    pd = np.flatnonzero(piv_a[:L] != piv_b[:L])
    if pd.size:
        cands.append(pd[0])
    if len(piv_a) != len(piv_b):
        cands.append(L)
    return int(min(cands)) if cands else -1
