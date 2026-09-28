"""Test matrix families. Everything is built in float64 with columns of norm <= 1;
walks cast to their dtype on entry, so every implementation and precision sees
the same B."""
import zlib
from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    family: str
    n: int
    params: dict
    B: np.ndarray                   # m x n, float64
    X: np.ndarray | None = None     # Harshaw covariates, n x d, rows scaled to max norm 1
    phi: float | None = None


def config_rng(seed, *key):
    return np.random.default_rng([seed, zlib.crc32(repr(key).encode())])


def _haar(n, rng):
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    return Q * np.sign(np.diag(R))


def _scale_cols(B):
    """Divide by the largest column norm (a single scalar, so kappa is unchanged)."""
    return B / np.linalg.norm(B, axis=0).max()


def gaussian(n, rng):
    return _scale_cols(rng.standard_normal((n, n)))


def conditioned(n, kappa, rng):
    s = np.logspace(0, -np.log10(kappa), n)
    return _scale_cols((_haar(n, rng) * s) @ _haar(n, rng).T)


def collinear(n, eta, rng):
    """n/2 pairs (v_j, (v_j + eta g_j)/||.||) of unit columns, interleaved."""
    assert n % 2 == 0
    v = rng.standard_normal((n, n // 2))
    v /= np.linalg.norm(v, axis=0)
    g = rng.standard_normal((n, n // 2))
    g /= np.linalg.norm(g, axis=0)
    w = v + eta * g
    w /= np.linalg.norm(w, axis=0)
    B = np.empty((n, n))
    B[:, 0::2] = v
    B[:, 1::2] = w
    return B


def harshaw_X(n, d, correlated, rng):
    """Gaussian covariates; `correlated` gives column singular values spread over 1e4."""
    X = rng.standard_normal((n, d))
    if correlated:
        X = (X * np.logspace(0, -4, d)) @ _haar(d, rng).T
    return X / np.linalg.norm(X, axis=1).max()


def harshaw_B(X, phi):
    """Augmented design matrix [sqrt(phi) I ; sqrt(1 - phi) X^T] of Harshaw et al."""
    n = X.shape[0]
    return np.vstack([np.sqrt(phi) * np.eye(n), np.sqrt(1 - phi) * X.T])


def make_instance(family, n, params, seed=0):
    rng = config_rng(seed, family, n, tuple(sorted(params.items())))
    if family == "gaussian":
        return Instance(family, n, params, gaussian(n, rng))
    if family == "conditioned":
        return Instance(family, n, params, conditioned(n, params["kappa"], rng))
    if family == "collinear":
        return Instance(family, n, params, collinear(n, params["eta"], rng))
    if family == "harshaw":
        X = harshaw_X(n, params["d"], params["corr"], rng)
        return Instance(family, n, params, harshaw_B(X, params["phi"]), X, params["phi"])
    raise ValueError(family)
