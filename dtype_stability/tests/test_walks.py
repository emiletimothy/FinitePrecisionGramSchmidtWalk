import numpy as np
import pytest

from gsw_stability import matrices, walks

DTYPES = [np.float64, np.float32, np.float16]


def harshaw_instance(seed, n=30, d=4, phi=0.3):
    X = matrices.harshaw_X(n, d, False, np.random.default_rng(seed))
    return X, phi, matrices.harshaw_B(X, phi)


def run(impl, X, phi, B, U, dtype, **kw):
    if impl == "lstsq":
        return walks.walk_lstsq(B, U, dtype, **kw)
    if impl == "chol":
        return walks.walk_harshaw_cholesky(X, phi, U, dtype, **kw)
    if impl == "chol_rk5":
        return walks.walk_harshaw_cholesky(X, phi, U, dtype, refactor_every=5, **kw)
    return walks.walk_gs_compress(B, U, dtype, **kw)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("impl", ["lstsq", "chol", "chol_rk5", "gs"])
def test_terminates_on_signs(impl, dtype):
    n = 20
    X, phi, B = harshaw_instance(1, n=n)
    for r in range(3):
        res = run(impl, X, phi, B, walks.uniforms(0, r, n), dtype)
        assert res.failed is None, res.failed
        assert np.all(np.abs(res.z) == 1)
        assert res.steps <= (2 * n if impl == "gs" else n)
        assert np.all(res.hit_step >= 0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_lstsq_terminates_on_gaussian(dtype):
    n = 24
    B = matrices.gaussian(n, np.random.default_rng(2))
    res = walks.walk_lstsq(B, walks.uniforms(0, 0, n), dtype)
    assert res.failed is None and np.all(np.abs(res.z) == 1) and res.steps <= n


@pytest.mark.parametrize("seed", range(5))
def test_float64_implementations_agree_on_harshaw(seed):
    X, phi, B = harshaw_instance(seed)
    n = X.shape[0]
    U = walks.uniforms(7, seed, n)
    ref = walks.walk_lstsq(B, U, np.float64)
    for impl in ["chol", "chol_rk5", "gs"]:
        res = run(impl, X, phi, B, U, np.float64)
        assert res.failed is None
        np.testing.assert_array_equal(res.z, ref.z)
        np.testing.assert_array_equal(res.pivots, ref.pivots)
        assert np.linalg.norm(B @ res.z - B @ ref.z) <= 1e-10


def test_same_uniforms_give_same_path_across_precisions():
    n = 20
    B = matrices.gaussian(n, np.random.default_rng(3))
    U = walks.uniforms(11, 0, n)
    np.testing.assert_array_equal(U, walks.uniforms(11, 0, n))
    r64 = walks.walk_lstsq(B, U, np.float64)
    r32 = walks.walk_lstsq(B, U, np.float32)
    np.testing.assert_array_equal(r64.pivots, r32.pivots)
    np.testing.assert_array_equal(r64.hit_step, r32.hit_step)
    np.testing.assert_array_equal(r64.z, r32.z.astype(np.float64))
