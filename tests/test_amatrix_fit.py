"""(C) Amatrix.fit round-trip: recover a known damped-cosine kernel."""

import numpy as np

import gleqpy.memory.time as memt


def _damped_cosines(t, coeffs, decays, freqs):
    out = np.zeros_like(t)
    for c, d, w in zip(coeffs, decays, freqs):
        out += c * np.cos(w * t) * np.exp(-d * t)
    return out


def test_fit_recovers_known_kernel():
    t = np.linspace(0.0, 8.0, 400)
    coeffs = [50.0, 20.0]
    decays = [0.8, 2.5]
    freqs = [3.0, 10.0]
    Kt = _damped_cosines(t, coeffs, decays, freqs)

    A = memt.Amatrix()
    Kt_fit, c_fit, d_fit, w_fit = A.fit(
        t, Kt, nterm=2,
        coeffs_guess=[40.0, 25.0],
        decays_guess=[1.0, 2.0],
        freqs_guess=[2.5, 9.0],
        bounds=(0.0, np.inf),
    )

    # the reconstructed kernel should match the input closely
    np.testing.assert_allclose(Kt_fit, Kt, rtol=1e-3, atol=1e-2 * np.abs(Kt).max())
    assert np.isfinite(Kt_fit).all()
    assert len(c_fit) == 2 and len(d_fit) == 2 and len(w_fit) == 2


def test_fit_amplitude_is_normalized_at_zero():
    """fit() rescales coefficients so K_fit(0) == K(0)."""
    t = np.linspace(0.0, 5.0, 200)
    Kt = _damped_cosines(t, [30.0], [1.5], [5.0])

    A = memt.Amatrix()
    Kt_fit, *_ = A.fit(t, Kt, nterm=1,
                       coeffs_guess=[25.0], decays_guess=[1.0], freqs_guess=[4.0],
                       bounds=(0.0, np.inf))

    np.testing.assert_allclose(Kt_fit[0], Kt[0], rtol=1e-8)
