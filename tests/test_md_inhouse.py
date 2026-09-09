"""Test 1: in-house GLE integrator reproduces a saved short trajectory."""

import numpy as np
import pytest

import _helpers as H


@pytest.fixture(scope="module")
def golden():
    return np.load(H._data_path("md_inhouse.npz"))


def test_inhouse_md_matches_golden(golden):
    out = H.inhouse_md_run()

    assert out["pos"].shape == golden["pos"].shape
    assert out["vel"].shape == golden["vel"].shape

    # seeded legacy-RandomState stream is bit-stable across numpy versions;
    # a small tolerance guards only against BLAS/FP-ordering drift.
    np.testing.assert_allclose(out["pos"], golden["pos"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(out["vel"], golden["vel"], rtol=1e-6, atol=1e-9)


def test_inhouse_md_is_finite():
    out = H.inhouse_md_run()
    assert np.isfinite(out["pos"]).all()
    assert np.isfinite(out["vel"]).all()
