lammps
======

files
-----

`sim.py` - Run this first, it runs a short simulation of a SPC/E + ion water box and 
outputs positions/velocities/forces at a 5fs interval

`lammps_memory.ipynb` - Extracts the GLE memory kernel on the ion degrees of freedom
from the simulation output. Reads the positions/velocities/forces in `explicit_run.lammpstrj`,
builds a potential of mean force (using `ion_potentials.py`), computes the velocity and
force time-correlation functions, and solves the Volterra equation for K(t). The kernel is
then fit to a sum of exponentially-damped cosines and written to A-matrix form
(`K_cation.A` / `K_anion.A`).

`ion_potentials.py` - Interaction potentials/forces between monatomic ion pairs.

`ion_explicit.dat` - Datafile for 1 2.5nm 1M SPC/E + LiCl simulation.

`ion_implicit.dat` - Datafile for LiCl simulation in vacuum/implicit solvent.

`K_cation.A` / `K_anion.A` - Fitted memory kernels for the cation (Li+) and anion (Cl-),
stored in A-matrix form. These are output by `lammps_memory.ipynb`.

`lammps_gle.ipynb` - Runs GLE dynamics natively in LAMMPS (the reverse direction of
`lammps_memory.ipynb`), using the built-in `fix gle` (Ceriotti et al.) and `fix gld`
(Baczewski-Bond, Prony series) fixes on the implicit-solvent ion system (`ion_implicit.dat`).
