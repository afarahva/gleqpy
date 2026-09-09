tests
=====

Unit / regression tests for gleqpy. Written for `pytest`.

Setup
-----

`pytest` (and `ase`, needed by the ASE integrator test) are declared as a test
extra:

```
pip install -e .[test]
```

The reference values were generated in the **ase_2025** conda environment
(python 3.13, numpy 2.5, scipy 1.18, ase 3.29). See the project memory / repo
notes for that environment.

Running
-------

```
pytest tests/
```

What is covered
---------------

Regression ("golden") tests — lock current numerical behavior:

- `test_md_inhouse.py`  — short seeded GLE run with the in-house integrator
  (`gleqpy.md.dynamics.GLD`) vs a saved trajectory.
- `test_md_ase.py`      — short seeded GLE run with the ASE integrator
  (`gleqpy.ase.dynamics.GLD` + `Harmonic3D`) vs a saved trajectory.
- `test_memory_kernels.py` — memory-kernel estimators (`calc_tcf` +
  `calc_memory_{dtrapz,midpt,fft}`) on precomputed pos/vel/frc/acc/dpmf arrays
  vs saved kernels. Also guards the numpy-2 `K_0=None` branch fix.
- `test_projection.py`  — `BathProjection` memory kernel / spring constant from
  a synthetic Hessian vs saved values.

Correctness tests — check against analytic / independent references:

- `test_matrix_ops.py`  — `matrixexp` vs `scipy.linalg.expm`; `matrixexp_deriv`
  vs `A^{-1} exp(A t)`.
- `test_tcf.py`         — `calc_tcf` "scipy" vs "direct" modes and vs a manual
  autocorrelation.
- `test_amatrix_fit.py` — `Amatrix.fit` recovers a known damped-cosine kernel.

Reproducibility
---------------

All integrators draw noise from the legacy global `numpy.random` (RandomState)
stream, whose bit sequence is stable across numpy versions, so seeded runs are
reproducible. Tests use `numpy.testing.assert_allclose` with tight tolerances
for pure-numeric checks and looser ones (rtol 1e-6) for MD trajectories to
absorb BLAS/floating-point ordering differences across platforms.

Regenerating the golden data
----------------------------

If an *intended* change alters the numerics, regenerate the references and
review the diff before committing:

```
cd tests
python generate_goldens.py     # rewrites tests/data/*.npz
```

The scenario builders live in `_helpers.py` and are shared by the generator and
the tests, so a test always reconstructs exactly the scenario whose reference
was saved.
