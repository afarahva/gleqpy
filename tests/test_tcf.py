"""(B) Correctness of time-correlation function estimation."""

import numpy as np

import gleqpy.memory.time as memt


def _manual_autocorr(x, max_t):
    """Reference (unbiased) autocorrelation of a 1D signal, matching calc_tcf."""
    nt = len(x)
    out = np.empty(max_t)
    for t in range(max_t):
        out[t] = np.sum(x[t:] * x[:nt - t]) / (nt - t)
    return out


def test_scipy_and_direct_modes_agree():
    rng = np.random.RandomState(0)
    x = rng.normal(size=(500, 2))
    max_t = 50

    tcf_scipy = memt.calc_tcf(x, x, max_t=max_t, mode="scipy")
    tcf_direct = memt.calc_tcf(x, x, max_t=max_t, mode="direct")

    assert tcf_scipy.shape == (max_t, 2)
    np.testing.assert_allclose(tcf_scipy, tcf_direct, rtol=1e-8, atol=1e-10)


def test_tcf_matches_manual_autocorrelation():
    rng = np.random.RandomState(1)
    x = rng.normal(size=(400, 1))
    max_t = 40

    tcf = memt.calc_tcf(x, x, max_t=max_t, mode="direct")[:, 0]
    ref = _manual_autocorr(x[:, 0], max_t)

    np.testing.assert_allclose(tcf, ref, rtol=1e-10, atol=1e-12)


def test_tcf_zero_lag_is_mean_square():
    rng = np.random.RandomState(2)
    x = rng.normal(size=(300, 3))

    tcf0 = memt.calc_tcf(x, x, max_t=5, mode="direct")[0]
    np.testing.assert_allclose(tcf0, np.mean(x ** 2, axis=0), rtol=1e-10)
