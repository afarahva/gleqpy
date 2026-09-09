"""
Shared, deterministic scenario builders for the gleqpy test suite.

Both ``generate_goldens.py`` and the tests import these builders so that a test
reconstructs *exactly* the scenario whose reference output was saved, and only
the numerical results live in ``tests/data/*.npz``.

Every builder seeds ``numpy.random`` internally.  All integrators in gleqpy draw
noise from the legacy global ``numpy.random`` (RandomState) stream, whose bit
sequence numpy guarantees to be stable across versions, so seeded runs are
reproducible on any numpy 1.x/2.x.
"""

import os
import numpy as np

SEED = 12345
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _data_path(name):
    return os.path.join(DATA_DIR, name)


# --------------------------------------------------------------------------- #
# 1) In-house MD: short seeded GLE run with a simple memory kernel
# --------------------------------------------------------------------------- #
def inhouse_md_run():
    """Run a short (100-step) in-house GLE simulation with a fixed seed.

    Returns dict with position/velocity trajectories (nframe, nsys, ndim).
    """
    import gleqpy.md.dynamics as md
    import gleqpy.md.forcefield as FFgen

    np.random.seed(SEED)

    # System / bath parameters (a single damped-oscillator memory kernel)
    nsys, ndim, m = 2, 1, 195.084
    frc_k = 30000.0
    x0 = np.zeros((nsys, ndim))
    temp = 2.4943
    md.kb = 1.0
    dt, nsteps = 0.001, 100

    Amat = np.array([[0.0, -10.0, -10.0],
                     [10.0,  0.5,   5.0],
                     [10.0, -5.0,   0.0]])
    As = Amat[1:, 1:]
    Bs = np.sqrt(md.kb * temp * (As + As.T))
    Asv = Amat[1:, 0:1]
    Avs = Amat[0:1, 1:]

    ff = FFgen.ff_harm(nsys, ndim, frc_k, x0)
    system = md.System(m, nsys=nsys, ndim=ndim, box_dim=1000)
    system.pos = np.random.normal(loc=x0, scale=np.sqrt(md.kb * temp / frc_k),
                                  size=(nsys, ndim))
    system.vel = np.random.normal(loc=0.0, scale=np.sqrt(md.kb * temp / m),
                                  size=(nsys, ndim))

    integrator = md.GLD(system, ff, dt, temp, As, Avs, Asv, Bs, PBC=False)
    report = md.reporter_PosVelAccFrc()
    integrator.reporters = [report]
    integrator.reportints = [1]
    integrator.run(nsteps)

    pos, vel, acc, frc = report.output()
    return {"pos": np.asarray(pos), "vel": np.asarray(vel)}


# --------------------------------------------------------------------------- #
# 2) ASE MD: short seeded GLE run with the ASE-coupled integrator
# --------------------------------------------------------------------------- #
def ase_md_run():
    """Run a short (100-step) ASE GLE simulation with a fixed seed.

    Returns dict with per-step positions/momenta of the single mobile atom.
    """
    from ase import units
    from ase.build import fcc111
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from gleqpy.ase.forcefield import Harmonic3D
    from gleqpy.ase.dynamics import GLD

    np.random.seed(SEED)

    temp, dt, nsteps = 300.0, 1.0, 100
    units_ps = units.fs * 1e3

    # simple isotropic 3x3 aux memory kernel (A-matrix), in ase units
    Amat = np.array([[0.0, -8.0, -8.0],
                     [8.0,  1.0,  4.0],
                     [8.0, -4.0,  0.0]]) / units_ps

    atoms = fcc111("Pt", size=(1, 1, 1))
    atoms.center(axis=2, vacuum=10)
    atoms.translate([0, 0, -10])
    atoms.pbc = (True, True, False)

    frc_k = np.diag([100.0, 100.0, 300.0]) * (units.kJ / units._Nav)
    atoms.calc = Harmonic3D(frc_k, atoms.positions.copy(),
                            np.diag(atoms.get_cell()))

    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)

    dyn = GLD(atoms, dt * units.fs, Amat, Amat_units="ase",
              temperature_K=temp, int_type=1)

    traj_pos, traj_mom = [], []

    def _record():
        traj_pos.append(atoms.get_positions().copy())
        traj_mom.append(atoms.get_momenta().copy())

    dyn.attach(_record, interval=1)
    dyn.run(nsteps)

    return {"pos": np.asarray(traj_pos), "mom": np.asarray(traj_mom)}


# --------------------------------------------------------------------------- #
# 3) Memory kernels from precomputed pos/vel/frc/acc/dpmf arrays
# --------------------------------------------------------------------------- #
def make_memory_input_arrays():
    """Generate a longer seeded in-house GLE trajectory for kernel extraction.

    Returns dict of pos/vel/frc/acc/dpmf (nframe, ndof) plus scalars needed by
    ``compute_memory_kernels``.  This is what gets *saved* as the precomputed
    fixture; the test then reads it back and runs the kernel estimators on it.
    """
    import gleqpy.md.dynamics as md
    import gleqpy.md.forcefield as FFgen

    np.random.seed(SEED)

    nsys, ndim, m = 4, 1, 195.084
    frc_k = 30000.0
    x0 = np.zeros((nsys, ndim))
    temp = 2.4943
    md.kb = 1.0
    dt, eq_steps, run_steps, stride = 0.001, 2000, 20000, 5

    Amat = np.array([[0.0, -10.0, -10.0],
                     [10.0,  0.5,   5.0],
                     [10.0, -5.0,   0.0]])
    As = Amat[1:, 1:]
    Bs = np.sqrt(md.kb * temp * (As + As.T))
    Asv = Amat[1:, 0:1]
    Avs = Amat[0:1, 1:]

    ff = FFgen.ff_harm(nsys, ndim, frc_k, x0)
    system = md.System(m, nsys=nsys, ndim=ndim, box_dim=1000)
    system.pos = np.random.normal(loc=x0, scale=np.sqrt(md.kb * temp / frc_k),
                                  size=(nsys, ndim))
    system.vel = np.random.normal(loc=0.0, scale=np.sqrt(md.kb * temp / m),
                                  size=(nsys, ndim))

    integrator = md.GLD(system, ff, dt, temp, As, Avs, Asv, Bs, PBC=False)
    integrator.run(eq_steps)

    report = md.reporter_PosVelAccFrc()
    integrator.reporters = [report]
    integrator.reportints = [stride]
    integrator.run(run_steps)

    pos, vel, acc, frc = map(np.asarray, report.output())

    # harmonic PMF force from the mean position (as in the notebooks)
    pos_mean = np.mean(pos)
    dpmf_func = FFgen.ff_harm(nsys, ndim, frc_k, pos_mean)
    dpmf = dpmf_func.calc_frc(pos)

    pos = pos.reshape(-1, nsys * ndim)
    vel = vel.reshape(-1, nsys * ndim)
    frc = frc.reshape(-1, nsys * ndim)
    acc = acc.reshape(-1, nsys * ndim)
    dpmf = dpmf.reshape(-1, nsys * ndim)

    return {"pos": pos, "vel": vel, "frc": frc, "acc": acc, "dpmf": dpmf,
            "m": np.array(m), "dt": np.array(dt * stride),
            "tcorr": np.array(300)}


def compute_memory_kernels(arrays):
    """Estimate memory kernels from precomputed pos/vel/frc/acc/dpmf arrays.

    Kept identical between generator and test so the test exercises the real
    ``calc_tcf`` / ``calc_memory_*`` code paths and compares to the saved values.
    """
    import gleqpy.memory.time as memt

    vel = arrays["vel"]
    acc = arrays["acc"]
    dpmf = arrays["dpmf"]
    m = float(arrays["m"])
    dt = float(arrays["dt"])
    tcorr = int(arrays["tcorr"])

    vel_tcf = memt.calc_tcf(vel, vel, max_t=tcorr, mode="scipy").mean(axis=1)
    dvel_tcf = -memt.calc_tcf(vel, acc, max_t=tcorr, mode="scipy").mean(axis=1)
    frc_tcf = memt.calc_tcf(acc - dpmf / m, vel, max_t=tcorr, mode="scipy").mean(axis=1)
    dfrc_tcf = -memt.calc_tcf(acc - dpmf / m, acc, max_t=tcorr, mode="scipy").mean(axis=1)

    Ktrap = memt.calc_memory_dtrapz(dvel_tcf, dfrc_tcf, vel_tcf[0], dt)
    Kmid = memt.calc_memory_midpt(vel_tcf, frc_tcf, dt)
    Kfft = np.real(memt.calc_memory_fft(vel_tcf, frc_tcf, dt))

    return {"Ktrap": Ktrap, "Kmid": Kmid, "Kfft": Kfft}


# --------------------------------------------------------------------------- #
# 5) Projection-operator memory kernel from a synthetic Hessian
# --------------------------------------------------------------------------- #
def projection_case():
    """Build a deterministic SPD Hessian and project out the bath.

    Returns the inputs (hess, masses, indx_P, t_arr) and outputs (Kt, springk).
    """
    from gleqpy.memory.proj import BathProjection

    rng = np.random.RandomState(SEED)
    natom = 4
    ndof = natom * 3

    # symmetric, diagonally dominant -> positive definite Hessian
    M = rng.normal(size=(ndof, ndof))
    hess = 0.5 * (M + M.T)
    hess += np.diag(np.full(ndof, ndof + 5.0))

    masses = np.full(natom, 195.084)  # per-atom masses
    indx_P = np.arange(3)             # one "surface site" = 3 dof
    t_arr = np.arange(200) * 0.025

    po = BathProjection(hess, masses, indx_P, remove_ifreq=True)
    Kt = po.calc_memory(t_arr)
    springk = po.calc_springk()

    return {"hess": hess, "masses": masses, "indx_P": indx_P, "t_arr": t_arr,
            "Kt": np.asarray(Kt), "springk": np.asarray(springk)}
