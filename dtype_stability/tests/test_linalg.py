import numpy as np
import pytest
import scipy.linalg as sla

from gsw_stability import linalg_lowp as la


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def spd(rng, n):
    G = rng.standard_normal((n, n))
    return G @ G.T + n * np.eye(n)


def test_tree_sum_matches_numpy_in_float64(rng):
    A = rng.standard_normal((37, 5))
    np.testing.assert_allclose(la.tree_sum(A, 0), A.sum(0), rtol=1e-13)
    np.testing.assert_allclose(la.tree_sum(A, 1), A.sum(1), rtol=1e-13)
    assert la.tree_sum(np.zeros((0, 3))).shape == (3,)


def test_float16_reductions_round_every_add():
    x = np.array([2048, 1, 1], dtype=np.float16)
    # float16 spacing at 2048 is 2, so 2048 + 1 rounds back to 2048 (ties to even).
    assert la.tree_sum(x) == np.float16(2048)
    assert la.tree_sum(x).dtype == np.float16
    # numpy's own float16 sum accumulates in float32 and gets 2050.
    assert np.sum(x) == np.float16(2050)
    assert la.dot(x, np.ones(3, np.float16)) == np.float16(2048)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_outputs_stay_in_dtype(rng, dtype):
    A = rng.standard_normal((12, 5)).astype(dtype) / 4
    b = rng.standard_normal(12).astype(dtype) / 4
    assert la.qr_lstsq(A, b).dtype == dtype
    M = spd(rng, 5).astype(dtype) / 10
    U = la.cholesky(M)
    assert U.dtype == dtype
    assert la.chol_rank1_update(U, b[:5]).dtype == dtype
    assert la.chol_solve(U, b[:5]).dtype == dtype


@pytest.mark.parametrize("shape", [(30, 12), (15, 15)])
def test_householder_qr_matches_scipy(rng, shape):
    A = rng.standard_normal(shape)
    k = shape[1]
    QR, tau = la.householder_qr(A)
    R = np.triu(QR[:k])
    Rs = sla.qr(A, mode="r")[0][:k]
    signs = np.sign(np.diag(R)) * np.sign(np.diag(Rs))
    np.testing.assert_allclose(R, signs[:, None] * Rs, atol=1e-12)
    QtA = np.column_stack([la.apply_qt(QR, tau, A[:, j]) for j in range(k)])
    np.testing.assert_allclose(QtA[:k], R, atol=1e-12)
    np.testing.assert_allclose(QtA[k:], 0, atol=1e-12)


@pytest.mark.parametrize("shape", [(40, 15), (20, 20), (9, 1)])
def test_qr_lstsq_matches_scipy(rng, shape):
    A = rng.standard_normal(shape)
    b = rng.standard_normal(shape[0])
    np.testing.assert_allclose(la.qr_lstsq(A, b), sla.lstsq(A, b)[0], rtol=1e-10, atol=1e-12)


def test_cholesky_matches_scipy(rng):
    M = spd(rng, 12)
    np.testing.assert_allclose(la.cholesky(M), sla.cholesky(M), rtol=1e-12)
    b = rng.standard_normal(12)
    np.testing.assert_allclose(la.chol_solve(la.cholesky(M), b), np.linalg.solve(M, b), rtol=1e-10)
    np.testing.assert_allclose(la.chol_inverse(M), np.linalg.inv(M), rtol=1e-10, atol=1e-14)
    with pytest.raises(la.BreakdownError):
        la.cholesky(-M)


def test_rank1_update_and_downdate_match_refactor(rng):
    M = spd(rng, 10)
    x = rng.standard_normal(10)
    U = la.cholesky(M)
    la.chol_rank1_update(U, x)
    np.testing.assert_allclose(U, sla.cholesky(M + np.outer(x, x)), rtol=1e-11, atol=1e-12)
    la.chol_rank1_downdate(U, x)
    np.testing.assert_allclose(U, sla.cholesky(M), rtol=1e-10, atol=1e-11)


def test_downdate_breakdown_raises(rng):
    U = la.cholesky(np.eye(4))
    with pytest.raises(la.BreakdownError):
        la.chol_rank1_downdate(U, np.array([2.0, 0, 0, 0]))


def test_givens_extreme_scale():
    for f, g in [(3.0, 4.0), (3e-200, 4e-200), (3e200, 4e200), (-3.0, 4.0)]:
        c, s, r = la.givens(np.float64(f), np.float64(g))
        np.testing.assert_allclose([c * f + s * g, -s * f + c * g], [r, 0], atol=abs(r) * 1e-15)
    c, s, r = la.givens(np.float16(3), np.float16(4))
    assert r.dtype == np.float16 and r == 5


def test_sm_remove_matches_direct_inverse(rng):
    Q = spd(rng, 9)
    C = np.linalg.inv(Q)
    pos = 4
    keep = np.arange(9) != pos
    got = la.sm_remove(C, pos, Q[keep, pos], Q[pos, pos])
    np.testing.assert_allclose(got, np.linalg.inv(Q[np.ix_(keep, keep)]), rtol=1e-10, atol=1e-13)
