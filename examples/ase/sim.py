#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLEPy
============

file: sim.py
description: Run ase simulation of a FCC Pt surface slab.
"""
from ase import units, Atoms
from ase.build import bulk, surface, fcc111
from ase.constraints import FixAtoms

from ase.io import Trajectory
from ase.visualize import view

from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet

import sys
import numpy as np
import matplotlib.pyplot as plt

##########  NICE PLOTS  ###########
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Computer-Modern']
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] =  16
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.right'] = True
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 18
###################################

#%%
##### Set up simulation

kbT_eV = 0.02585  # 300K in eV

# Simulation parameters
nvt_steps = 25000     # Equilibriation Steps
nve_steps = 1000000   # Production Steps
interval  = 5         # Interval for printing positions
Temp      = 300       # Temperature

# Surface slab parameters
Nx, Ny, Nz = 3, 3, 5
lattice_ele = "Pt"

Nsurface = Nx * Ny
Nlattice = Nsurface * Nz

# Create system
atoms = fcc111('Pt', size=(Nx, Ny, Nz))
atoms.center(axis=2, vacuum = 10)
atoms.translate([0, 0, -10])

atoms.set_cell(atoms.get_cell())
atoms.pbc=(True, True, False)

# Constraint corners
indices_corners = [0,Nx-1,Nx*Ny-Nx,Nx*Ny-1]
constraints = [FixAtoms(indices=indices_corners)]
atoms.set_constraint(constraints)

# Set calculator (EMT)
try:
    from asap3 import EMT
    calc = EMT()
    calc.set_atoms(atoms)
    atoms.calc   = calc
    print("Using ASAP EMT calculator")
except:
    print("ASAP not installed defaulting to ase emt")
    from ase.calculators.emt import EMT
    calc = EMT
    calc.set_atoms(atoms)
    atoms.calc   = calc
#%%
##### Run Simulations

# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, temperature_K=Temp)

# Equilibrate Lattice using NVT
dyn_nvt = Langevin(atoms, 1*units.fs, temperature_K = Temp, friction=0.005)
dyn_nvt.attach(MDLogger(dyn_nvt, atoms, sys.stdout, header=True, mode="w"),
              interval=1000)
dyn_nvt.run(nvt_steps)

# MD run NVE
dyn = VelocityVerlet(atoms, 1 * units.fs)  # 1 fs time step.
dyn.attach(MDLogger(dyn, atoms, sys.stdout, header=True, mode="w"), 
          interval=1000)

traj = Trajectory('md.traj', 'w', atoms)
dyn.attach(traj.write, interval=interval)
dyn.run(nve_steps)

#%%
##### End of run trajectory i/o

# Load trajectory
traj_sim = Trajectory("md.traj", 'r')

pos_array = []
vel_array = []
frc_array = []

indx_lo = (Nz-1) * Nsurface
indx_hi = (Nz) * Nsurface

for atoms in traj_sim:
    pos = atoms.get_positions()[indx_lo:indx_hi]
    vel = atoms.get_velocities()[indx_lo:indx_hi]
    frc = atoms.get_forces()[indx_lo:indx_hi]
    
    pos_array.append(pos)
    vel_array.append(vel)
    frc_array.append(frc)

# Convert units
pos_array = np.array(pos_array) / (units.nm)
vel_array = np.array(vel_array) / (units.nm / (units.fs * 1e3) )
frc_array = np.array(frc_array) / (units.kJ/units._Nav/units.nm)

# save to npz fi;e
np.savez("traj.npz", pos=pos_array, vel=vel_array, frc=frc_array)
