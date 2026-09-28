"""Per-step comparison of a walk against a float64 solve from the same state.

The reference at step t uses the walk's own (z_t, alive set, pivot), upcast to
float64, and the dtype-rounded B upcast to float64, so it measures arithmetic
error in the step rather than the representation error of B. For the Cholesky
and GS cores the walk's inputs are the rounded X or Q rather than the rounded B,
so their error floor includes an O(eps) mismatch between those roundings.
"""
import numpy as np
import scipy.linalg as sla

from .walks import ratio_test

FIELDS = ("run", "t", "k", "kappa", "err_Bu", "err_u", "norm_u", "norm_Bu",
          "err_p", "drift", "inv_drift")


class StepLogger:
    def __init__(self, B_dtype, run):
        self.B = np.asarray(B_dtype, dtype=np.float64)
        self.run = run
        self.rows = []

    def __call__(self, t, z_before, p, idx, u, prob, core, active):
        B = self.B
        n = B.shape[1]
        u64 = np.zeros(n)
        u64[p] = 1.0
        kappa = 1.0
        if len(idx):
            Bs = B[:, idx]
            u64[idx] = -sla.lstsq(Bs, B[:, p])[0]
            s = sla.svdvals(Bs)
            kappa = s[0] / s[-1] if s[-1] > 0 else np.inf
        du = u.astype(np.float64) - u64
        dp, dm, _, _ = ratio_test(z_before.astype(np.float64), u64, idx, p,
                                  10 * np.finfo(np.float64).eps)
        p64 = dm / (dp + dm)

        drift = inv_drift = np.nan
        if hasattr(core, "U"):
            members = active.copy()
            members[p] = False
            Xs = core.Xt.astype(np.float64)[:, members]
            M = float(core.c) * np.eye(Xs.shape[0]) + Xs @ Xs.T
            U64 = core.U.astype(np.float64)
            drift = np.linalg.norm(U64.T @ U64 - M) / np.linalg.norm(M)
        if hasattr(core, "C") and len(core.idx):
            Q = core.Q.astype(np.float64)[np.ix_(core.idx, core.idx)]
            k = len(core.idx)
            inv_drift = np.linalg.norm(core.C.astype(np.float64) @ Q - np.eye(k)) / np.sqrt(k)

        self.rows.append((self.run, t, len(idx), kappa, np.linalg.norm(B @ du),
                          np.max(np.abs(du)), np.max(np.abs(u64)), np.linalg.norm(B @ u64),
                          abs(float(prob) - p64), drift, inv_drift))

    def array(self):
        return np.array(self.rows, dtype=np.float64).reshape(-1, len(FIELDS))
