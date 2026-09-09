"""Test 2: ASE-coupled GLE integrator reproduces a saved short trajectory."""

import numpy as np
import pytest

import _helpers as H

pytest.importorskip("ase", reason="ASE not installed")


@pytest.fixture(scope="module")
def golden():
    return np.load(H._data_path("md_ase.npz"))


def test_ase_md_matches_golden(golden):
    out = H.ase_md_run()

    assert out["pos"].shape == golden["pos"].shape
    assert out["mom"].shape == golden["mom"].shape

    np.testing.assert_allclose(out["pos"], golden["pos"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(out["mom"], golden["mom"], rtol=1e-6, atol=1e-9)


def test_ase_md_is_finite():
    out = H.ase_md_run()
    assert np.isfinite(out["pos"]).all()
    assert np.isfinite(out["mom"]).all()
