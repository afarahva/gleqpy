lammps
======

files
-----

`sim.py` - Run this first, it runs a short simulation of a SPC/E + ion water box and 
outputs positions/velocities/forces at a 5fs interval

`ase_memory.ipynb` - Analyzes the output of `sim.py` and calculates a memory kernel 
(`Kz_5term.A`)and a harmonic force constant `frck.dat` for surface site fluctuations. 

`ase_gle.ipynb` - Uses `Kz_5term.A` and `frck.dat` to run a GLE simulation for a Pt(111)
surface site. 