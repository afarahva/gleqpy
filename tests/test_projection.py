"""(E) Projection-operator memory kernel from a synthetic Hessian."""

import numpy as np
import pytest

from gleqpy.memory.proj import BathProjection
import _helpers as H


@pytest.fixture(scope="module")
def golden():
    d = np.load(H._data_path("projection.npz"))
    return {k: d[k] for k in d.files}


def test_projection_matches_golden(golden):
    po = BathProjection(golden["hess"], golden["masses"], golden["indx_P"],
                        remove_ifreq=True)
    Kt = po.calc_memory(golden["t_arr"])
    springk = po.calc_springk()

    np.testing.assert_allclose(Kt, golden["Kt"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(springk, golden["springk"], rtol=1e-8, atol=1e-8)


def test_projection_kernel_properties(golden):
    po = BathProjection(golden["hess"], golden["masses"], golden["indx_P"])
    Kt = po.calc_memory(golden["t_arr"])

    # shape is (nt, nsys, nsys) with nsys = len(indx_P) = 3
    assert Kt.shape == (len(golden["t_arr"]), 3, 3)
    assert np.isfinite(Kt).all()

    # K(t=0) should be symmetric for this symmetric-Hessian system
    np.testing.assert_allclose(Kt[0], Kt[0].T, rtol=1e-8, atol=1e-8)


def test_springk_is_symmetric(golden):
    po = BathProjection(golden["hess"], golden["masses"], golden["indx_P"])
    springk = po.calc_springk()
    np.testing.assert_allclose(springk, springk.T, rtol=1e-8, atol=1e-8)
