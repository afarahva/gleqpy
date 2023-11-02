ase
===

files
-----

`sim.py` - Run this first, it runs a short simulation of a Pt(111) surface slab and
outputs the positions/velocities/forces at a 5fs interval.

`lammps_memory.ipynb` - Analyzes the output of `sim.py` and calculates a memory kernels 
for cations (`K_cation.A`) and anions (`K_anion.A`)

`lammps_gle.ipynb` - Runs a GLE simulation of ions using data from `K_cation.A` and 
`K_anion.A`

`ion_potentials.py` - Interaction potentials/forces between monatomic ion pairs.

`ion_explicit.dat` - Datafile for 1 2.5nm 1M SPC/E + LiCl simulation.

`ion_implicit.dat` - Datafile for LiCl simulation in vacuum/implicit solvent.