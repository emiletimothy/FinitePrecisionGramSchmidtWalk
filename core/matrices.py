"""Matrix families for the GSW experiments.

Every experiment takes --matrix NAME (plus --m where the family needs it).
Each family returns a nested column set — B at size n is the n-column prefix
of one underlying draw — so n-sweeps are not confounded by a redrawn matrix.
Deterministic families (identity, lower_bound) are rebuilt per n, which is
equivalent since there is nothing to confound.

    get_B = matrices.matrix_family("clustered", m=30, n_max=1600, seed=0)
    B = get_B(100)                      # (30, 100), a prefix of the n_max set
"""

import numpy as np

MATRIX_NAMES = ("clustered", "gaussian", "higgs", "identity", "lower_bound")


def clustered(m: int = 30, n_max: int = 1600, seed: int = 0):
    """u ~ N(0,I_m) direction + (1/sqrt(m)) noise, column-normalised."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(m)
    u /= np.linalg.norm(u)
    noise = rng.standard_normal((m, n_max))

    def get_B(n: int) -> np.ndarray:
        B = u[:, None] + noise[:, :n] / np.sqrt(m)
        return B / np.linalg.norm(B, axis=0)

    return get_B


def gaussian(m: int = 30, n_max: int = 1600, seed: int = 0):
    """iid N(0,1) entries, column-normalised."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((m, n_max))

    def get_B(n: int) -> np.ndarray:
        B = W[:, :n].copy()
        return B / np.linalg.norm(B, axis=0)

    return get_B


def higgs(m: int | None = None, n_max: int = 1600, seed: int = 0):
    """Two low-level angular-momentum Higgs features (m = 2 fixed).

    From OpenML data_id 23512 — the two-sample experiment of Low-Rank
    Thinning (Liu et al. 2020 / Domingo-Enrich et al. 2023, Sec. 6.2).
    """
    from sklearn.datasets import fetch_openml

    print(f"Loading Higgs from OpenML (data_id=23512), n_max={n_max}...")
    X, _ = fetch_openml(
        data_id=23512, as_frame=False, return_X_y=True, parser="liac-arff",
    )
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_max, len(X)), replace=False)
    B = X[idx][:, [0, 1]].T  # (2, n_max)
    norms = np.linalg.norm(B, axis=0)
    B = B / np.where(norms == 0, 1.0, norms)
    return lambda n: B[:, :n]


def identity(m: int | None = None, n_max: int | None = None, seed: int = 0):
    """B = I_n — the tight Rademacher instance (m = n)."""
    return lambda n: np.eye(n)


def lower_bound_matrix(n: int) -> np.ndarray:
    """B in R^{(n+1) x n}, columns v_i = (e_0 + e_i)/sqrt(2) — Theorem 4.1."""
    B = np.zeros((n + 1, n))
    c = 1.0 / np.sqrt(2.0)
    B[0, :] = c
    B[1:, :] = np.eye(n) * c
    return B


def lower_bound(m: int | None = None, n_max: int | None = None, seed: int = 0):
    """Theorem-4.1 adversarial matrix (m = n + 1)."""
    return lower_bound_matrix


def matrix_family(name: str, m: int = 30, n_max: int = 1600, seed: int = 0):
    """Return get_B(n) -> (m_n, n) column-normalised B for the named family.

    `m` is only used by the randomized families that need a row count
    (clustered, gaussian); higgs/identity/lower_bound determine their own.
    """
    builders = {
        "clustered": clustered,
        "gaussian": gaussian,
        "higgs": higgs,
        "identity": identity,
        "lower_bound": lower_bound,
    }
    if name not in builders:
        raise ValueError(f"unknown matrix {name!r}; choose from {MATRIX_NAMES}")
    return builders[name](m=m, n_max=n_max, seed=seed)


def add_matrix_arguments(parser, default: str = "higgs") -> None:
    """Standard --matrix/--m CLI pair used by every experiment script."""
    parser.add_argument("--matrix", choices=MATRIX_NAMES, default=default,
                        help="B family (default %(default)s)")
    parser.add_argument("--m", type=int, default=30,
                        help="rows for clustered/gaussian (default 30)")
