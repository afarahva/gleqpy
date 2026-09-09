"""Test 3: memory-kernel estimators on precomputed pos/vel/frc/acc/dpmf arrays.

Also includes a regression guard (D) for the numpy-2 ``K_0=None`` branch bug.
"""

import numpy as np
import pytest

import gleqpy.memory.time as memt
import _helpers as H


@pytest.fixture(scope="module")
def arrays():
    d = np.load(H._data_path("memory_arrays.npz"))
    return {k: d[k] for k in d.files}


def test_memory_kernels_match_golden(arrays):
    kernels = H.compute_memory_kernels(arrays)

    np.testing.assert_allclose(kernels["Ktrap"], arrays["expected_Ktrap"],
                               rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(kernels["Kmid"], arrays["expected_Kmid"],
                               rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(kernels["Kfft"], arrays["expected_Kfft"],
                               rtol=1e-8, atol=1e-8)


def test_memory_kernels_finite(arrays):
    kernels = H.compute_memory_kernels(arrays)
    for name, K in kernels.items():
        assert np.isfinite(K).all(), f"{name} contains non-finite values"


# --- (D) regression guard for the numpy-2 `np.all(None)` branch bug --------- #
# Before the fix, `if np.all(K_0) != None:` took the wrong branch under numpy 2
# and set K[0]=None -> the whole kernel became NaN.

def test_dtrapz_default_K0_is_finite(arrays):
    """calc_memory_dtrapz with default K_0=None must return a finite kernel."""
    vel = arrays["vel"]; acc = arrays["acc"]; dpmf = arrays["dpmf"]
    m = float(arrays["m"]); dt = float(arrays["dt"]); tcorr = int(arrays["tcorr"])

    vel_tcf = memt.calc_tcf(vel, vel, max_t=tcorr).mean(axis=1)
    dvel_tcf = -memt.calc_tcf(vel, acc, max_t=tcorr).mean(axis=1)
    dfrc_tcf = -memt.calc_tcf(acc - dpmf / m, acc, max_t=tcorr).mean(axis=1)

    K = memt.calc_memory_dtrapz(dvel_tcf, dfrc_tcf, vel_tcf[0], dt)  # K_0=None
    assert np.isfinite(K).all()
    assert not np.isnan(K[0])


def test_midpt_default_K0_is_finite(arrays):
    """calc_memory_midpt with default K_0=None must return a finite kernel."""
    vel = arrays["vel"]; acc = arrays["acc"]; dpmf = arrays["dpmf"]
    m = float(arrays["m"]); dt = float(arrays["dt"]); tcorr = int(arrays["tcorr"])

    vel_tcf = memt.calc_tcf(vel, vel, max_t=tcorr).mean(axis=1)
    frc_tcf = memt.calc_tcf(acc - dpmf / m, vel, max_t=tcorr).mean(axis=1)

    K = memt.calc_memory_midpt(vel_tcf, frc_tcf, dt)  # K_0=None
    assert np.isfinite(K).all()


def test_dtrapz_respects_explicit_K0(arrays):
    """An explicit K_0 must actually be used as the initial value."""
    vel = arrays["vel"]; acc = arrays["acc"]; dpmf = arrays["dpmf"]
    m = float(arrays["m"]); dt = float(arrays["dt"]); tcorr = int(arrays["tcorr"])

    vel_tcf = memt.calc_tcf(vel, vel, max_t=tcorr).mean(axis=1)
    dvel_tcf = -memt.calc_tcf(vel, acc, max_t=tcorr).mean(axis=1)
    dfrc_tcf = -memt.calc_tcf(acc - dpmf / m, acc, max_t=tcorr).mean(axis=1)

    K = memt.calc_memory_dtrapz(dvel_tcf, dfrc_tcf, vel_tcf[0], dt, K_0=0.0)
    assert K[0] == 0.0
