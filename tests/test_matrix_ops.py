"""(A) Correctness of the matrix-exponential helpers vs scipy / finite diff."""

import numpy as np
import pytest
from scipy.linalg import expm

import gleqpy.memory.time as memt


def test_matrixexp_hermitian_matches_scipy():
    rng = np.random.RandomState(0)
    n = 4
    M = rng.normal(size=(n, n))
    A = 0.5 * (M + M.T)                      # symmetric (Hermitian)
    t_arr = np.linspace(0.0, 2.0, 7)

    mexp = memt.matrixexp(A, t_arr, Hermitian=True)
    assert mexp.shape == (len(t_arr), n, n)
    for i, t in enumerate(t_arr):
        np.testing.assert_allclose(np.real(mexp[i]), expm(A * t),
                                   rtol=1e-9, atol=1e-11)


def test_matrixexp_general_matches_scipy():
    rng = np.random.RandomState(1)
    n = 3
    A = rng.normal(size=(n, n))             # non-symmetric
    t_arr = np.linspace(0.0, 1.5, 5)

    mexp = memt.matrixexp(A, t_arr, Hermitian=False)
    for i, t in enumerate(t_arr):
        np.testing.assert_allclose(mexp[i], expm(A * t), rtol=1e-8, atol=1e-10)


def test_matrixexp_deriv_equals_Ainv_expm():
    """matrixexp_deriv puts exp(lambda*t)/lambda on the diagonal, i.e. it returns
    A^{-1} @ exp(A t) (an antiderivative-like quantity), NOT d/dt exp(A t).
    Locking that actual behavior against scipy.  (Name is misleading.)"""
    rng = np.random.RandomState(2)
    n = 3
    M = rng.normal(size=(n, n))
    A = 0.5 * (M + M.T) + np.diag(np.full(n, 4.0))  # SPD -> invertible, no zero eig
    t_arr = np.array([0.3, 0.7, 1.1])

    out = memt.matrixexp_deriv(A, t_arr, Hermitian=True)

    Ainv = np.linalg.inv(A)
    for i, t in enumerate(t_arr):
        np.testing.assert_allclose(np.real(out[i]), Ainv @ expm(A * t),
                                   rtol=1e-8, atol=1e-10)
