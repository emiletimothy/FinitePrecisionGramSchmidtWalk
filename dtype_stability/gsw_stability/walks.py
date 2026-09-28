"""The Gram-Schmidt walk with dtype-agnostic control flow.

All implementations share `_walk`, which owns the alive set, pivot choice,
ratio test, coin flip and snapping. They differ only in the `core` that computes
the step direction. z, u, the step sizes and the probability are all computed in
`dtype`; the only float64 values are the pre-drawn uniforms, which are the
randomness rather than arithmetic.

Core protocol:
    init(active, p)      first pivot chosen; set up state for active minus {p}
    remove(i)            i leaves the non-pivot active set (froze or became pivot)
    direction(idx, p)    u restricted to the non-pivot active coordinates idx
    end_step(t, active, p)
"""
import time
from dataclasses import dataclass

import numpy as np

from . import linalg_lowp as la


@dataclass
class WalkResult:
    z: np.ndarray
    steps: int
    failed: str | None
    pivots: np.ndarray      # pivot used at each step
    hit_step: np.ndarray    # step at which |z_i| first became 1, -1 if never
    forced_max: float       # max |z_i - target| over ratio-test coordinates before snapping
    tol_snaps: int          # other coordinates snapped because they were within tolerance
    tol_max: float
    clamps: int             # coordinates pushed back from outside [-1, 1]
    clamp_max: float
    zero_steps: int
    seconds: float


def uniforms(seed, run, n):
    """Per-run randomness: row t holds (pivot draw, coin draw) for step t."""
    return np.random.default_rng([seed, run]).random((2 * n + 4, 2))


def ratio_test(z, u, idx, p, zero_tol):
    """Largest steps (d+, d-) keeping z + d u in the cube, and the coordinates that
    attain each. Entries with |u_i| <= zero_tol are skipped, as in GSWDesign.jl."""
    dt = z.dtype
    ui, zi = u[idx], z[idx]
    ok = np.abs(ui) > zero_tol
    ui, zi, ii = ui[ok], zi[ok], idx[ok]
    s = np.sign(ui)
    one = dt.type(1)
    dp = np.concatenate([(s - zi) / ui, np.array([one - z[p]], dtype=dt)])
    dm = np.concatenate([(s + zi) / ui, np.array([one + z[p]], dtype=dt)])
    coords = np.concatenate([ii, [p]])
    dplus, dminus = dp.min(), dm.min()
    return dplus, dminus, coords[dp == dplus], coords[dm == dminus]


def _walk(core, n, U, dtype, z0=None, snap_c=10.0, logger=None, one_at_a_time=False,
          max_steps=None):
    dt = np.dtype(dtype)
    one = dt.type(1)
    eps = np.finfo(dt).eps
    zero_tol = dt.type(10 * eps)
    snap_tol = dt.type(snap_c * eps)

    z = np.zeros(n, dtype=dt) if z0 is None else np.asarray(z0, dtype=dt).copy()
    active = np.abs(z) < one
    hit_step = np.full(n, -1)
    pivots = []
    forced_max = tol_max = clamp_max = 0.0
    tol_snaps = clamps = zero_steps = 0
    failed = None
    p = -1
    t = 0
    t0 = time.perf_counter()
    with np.errstate(all="ignore"):
        try:
            while not np.all(np.abs(z) == one):
                if t >= U.shape[0]:
                    failed = "step_limit"
                    break
                if max_steps is not None and t >= max_steps:
                    failed = "max_steps"   # timing benchmarks only
                    break
                if p < 0 or not active[p]:
                    cand = np.flatnonzero(active)
                    newp = cand[min(int(U[t, 0] * len(cand)), len(cand) - 1)]
                    if p < 0:
                        p = newp
                        core.init(active, p)
                    else:
                        p = newp
                        core.remove(p)
                nonpiv = active.copy()
                nonpiv[p] = False
                idx = np.flatnonzero(nonpiv)

                u_np = core.direction(idx, p)
                if not np.all(np.isfinite(u_np)):
                    failed = "nonfinite_u"
                    break
                u = np.zeros(n, dtype=dt)
                u[idx] = u_np
                u[p] = one

                dplus, dminus, hp, hm = ratio_test(z, u, idx, p, zero_tol)
                denom = dplus + dminus
                prob = dminus / denom if denom != 0 else dt.type(np.nan)
                if denom == 0:
                    delta, hits, targets = dt.type(0), hp[:0], z[:0]
                elif not np.isfinite(prob):
                    failed = "nonfinite_p"
                    break
                elif U[t, 1] < float(prob):
                    delta, hits = dplus, hp
                    targets = np.sign(u[hits])
                else:
                    delta, hits = -dminus, hm
                    targets = -np.sign(u[hits])
                if delta == 0:
                    zero_steps += 1

                z_before = z.copy() if logger is not None else None
                z[active] = z[active] + delta * u[active]

                if hits.size:
                    forced_max = max(forced_max, float(np.max(np.abs(
                        z[hits].astype(np.float64) - targets.astype(np.float64)))))
                    z[hits] = targets
                az = np.abs(z)
                over = active & (az > one)
                near = active & (az < one) & (az >= one - snap_tol)
                if over.any():
                    clamps += int(over.sum())
                    clamp_max = max(clamp_max, float(np.max(az[over].astype(np.float64) - 1)))
                if near.any():
                    tol_snaps += int(near.sum())
                    tol_max = max(tol_max, float(np.max(1 - az[near].astype(np.float64))))
                z[over | near] = np.sign(z[over | near])

                at_bound = active & (np.abs(z) == one)
                hit_step[at_bound & (hit_step < 0)] = t
                pivots.append(p)

                removed = np.flatnonzero(at_bound)
                if one_at_a_time:
                    removed = removed[:1]
                for i in removed:
                    active[i] = False
                    if i != p:
                        core.remove(i)
                core.end_step(t, active, p)
                if logger is not None:
                    logger(t=t, z_before=z_before, p=p, idx=idx, u=u, prob=prob,
                           core=core, active=active)
                t += 1
        except la.BreakdownError as e:
            failed = "breakdown:" + str(e).split(":")[0]
    return WalkResult(z, t, failed, np.array(pivots, dtype=np.int64), hit_step,
                      forced_max, tol_snaps, tol_max, clamps, clamp_max, zero_steps,
                      time.perf_counter() - t0)


# ---------------------------------------------------------------- cores

class _Core:
    def init(self, active, p):
        pass

    def remove(self, i):
        pass

    def end_step(self, t, active, p):
        pass


class LstsqCore(_Core):
    """Fresh Householder least-squares solve every step."""

    def __init__(self, B):
        self.B = B

    def direction(self, idx, p):
        return -la.qr_lstsq(self.B[:, idx], self.B[:, p])


class HarshawCholeskyCore(_Core):
    """GSWDesign.jl: Cholesky factor of M = (phi/(1-phi)) I + X_S X_S^T over the
    non-pivot alive set S, downdated as units leave S."""

    def __init__(self, X, phi, dt, refactor_every=None):
        self.Xt = np.ascontiguousarray(np.asarray(X, dtype=np.float64).T).astype(dt)  # d x n
        one = dt.type(1)
        lam = dt.type(phi)
        self.c = lam / (one - lam)
        self.mult = (one - lam) / lam
        self.refactor_every = refactor_every
        self.dt = dt

    def _factor(self, members):
        # cholesky(c I) followed by one rank-one update per unit, as sample_gs_walk does.
        U = np.sqrt(self.c) * np.eye(self.Xt.shape[0], dtype=self.dt)
        for i in members:
            la.chol_rank1_update(U, self.Xt[:, i])
        return U

    def init(self, active, p):
        self.U = self._factor(range(self.Xt.shape[1]))
        self.remove(p)

    def remove(self, i):
        la.chol_rank1_downdate(self.U, self.Xt[:, i])

    def direction(self, idx, p):
        # compute_step_direction, operation for operation:
        #   a = L (U x_p) - c x_p;  a = M \ a;  a = ((1-phi)/phi) (a - x_p);  u = X^T a
        xp = self.Xt[:, p]
        a = la.matvec(self.U.T, la.matvec(self.U, xp)) - self.c * xp
        a = la.chol_solve(self.U, a)
        a = self.mult * (a - xp)
        return la.rmatvec(self.Xt[:, idx], a)

    def end_step(self, t, active, p):
        # Refactor from scratch: form M = c I + X_S X_S^T in dtype and factor it.
        if self.refactor_every and (t + 1) % self.refactor_every == 0:
            members = active.copy()
            members[p] = False
            M = la.gram(self.Xt[:, members].T)
            M.flat[::M.shape[0] + 1] += self.c
            self.U = la.cholesky(M)


class GSCubicCore(_Core):
    """Low-Rank Thinning, GS-Halve-Cubic: explicit C = (Q_{S,S})^{-1} with Q = B^T B,
    updated by block inversion + Sherman-Morrison as indices leave S."""

    def __init__(self, B):
        self.Q = la.gram(B)

    def init(self, active, p):
        idx = np.flatnonzero(active)
        self.idx = idx[idx != p]
        self.C = la.chol_inverse(self.Q[np.ix_(self.idx, self.idx)])

    def remove(self, i):
        pos = int(np.searchsorted(self.idx, i))
        keep = np.delete(self.idx, pos)
        self.C = la.sm_remove(self.C, pos, self.Q[keep, i], self.Q[i, i])
        self.idx = keep

    def direction(self, idx, p):
        assert np.array_equal(idx, self.idx)
        return -la.matvec(self.C, self.Q[idx, p])


# ---------------------------------------------------------------- public API

def walk_lstsq(B, U, dtype, **kw):
    """General walk, fresh least-squares solve every step (the reference)."""
    B = np.asarray(B).astype(dtype)
    return _walk(LstsqCore(B), B.shape[1], U, dtype, **kw)


def walk_harshaw_cholesky(X, phi, U, dtype, refactor_every=None, **kw):
    """Harshaw et al. design walk on B = [sqrt(phi) I; sqrt(1-phi) X^T]."""
    dt = np.dtype(dtype)
    return _walk(HarshawCholeskyCore(X, phi, dt, refactor_every), X.shape[0], U, dtype, **kw)


def walk_gs_compress(B, U, dtype, **kw):
    """kernel_gs_walk_cubic of Carrell et al. on Q = B^T B; removes one index per step."""
    B = np.asarray(B).astype(dtype)
    return _walk(GSCubicCore(B), B.shape[1], U, dtype, one_at_a_time=True, **kw)
