import numpy as np

from gsw_stability import metrics


def test_psi2_orlicz_of_standard_gaussian():
    # E exp(Y^2/K^2) = (1 - 2/K^2)^{-1/2} = 2  =>  K = sqrt(8/3)
    y = np.random.default_rng(0).standard_normal(200_000)
    assert abs(metrics.psi2_orlicz(y) - np.sqrt(8 / 3)) < 0.03
    assert metrics.psi2_orlicz(np.zeros(10)) == 0.0


def test_psi2_moment_of_standard_gaussian():
    y = np.random.default_rng(1).standard_normal(200_000)
    # (E Y^2)^{1/2}/sqrt(2) = 0.707 dominates (E Y^8)^{1/8}/sqrt(8) = 105^{1/8}/sqrt(8) = 0.63
    assert abs(metrics.psi2_moment(y) - 1 / np.sqrt(2)) < 0.01


def test_first_divergence_with_stored_int16_arrays():
    hit = np.array([0, 2, -1, 1], dtype=np.int16)
    piv = np.array([3, 3, 1], dtype=np.int16)
    assert metrics.first_divergence(hit, piv, hit.copy(), piv.copy()) == -1
    other = hit.copy()
    other[1] = 1
    assert metrics.first_divergence(hit, piv, other, piv) == 1
    assert metrics.first_divergence(hit, piv, hit, piv[:2]) == 2
