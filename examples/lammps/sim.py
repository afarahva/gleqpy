#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLEPy
============

file: sim.py
description: Run lammps simulation of a small SPC/E ion box. 
"""

import os
import sys

from lammps import lammps
from mpi4py import MPI

#%%
##### Script parameters
time_step = 1.0
npt_steps  = 50000
nvt_steps = 50000
nve_steps = 200000

#%%
##### Run simulation

lmp = lammps()
me = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()
print("Proc %d out of %d procs has" % (me,nprocs),lmp)

block = """

# LAMMPS script file to simulate equilibrated bulk-ion box

units       real
dimension   3
boundary    p p p

atom_style  full
pair_style	lj/cut/coul/long 12
bond_style	harmonic
angle_style	harmonic

read_data "ion_explicit.dat"

group cation type 1
group anion type 2
group ion type 1 2
group water type 3 4

pair_modify shift yes mix arithmetic
# i j epsilon sigma
pair_coeff 1 1 0.33673 1.4094	#Li+  
pair_coeff 2 2 0.01279 4.8305	#Cl-  
pair_coeff 3 3 0.1553 3.166    #SPC/E O
pair_coeff 4 4 0 10            #SPC/E H

# These bond and angle coefficients
# SPC/E requires the molecule geometry to be kept fixed, e.g. with SHAKE
bond_coeff 1 100 1
angle_coeff 1 300 109.47

kspace_style pppm 1e-6
dielectric   1.0

# Set up integration
timestep %0.1f # fs
run_style verlet
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

# write_dump all xyz init.xyz modify element Li Cl O H

##################
### Now do NPT ###
##################

# Constrain OH bond lengths and HOH angles
fix water_shake_1 water shake 1e-9 200 0 b 1 a 1

#temp tstart tstop tdamp iso pstart pstop pdamp
fix ensemble_npt all npt temp 300 300 200 iso 1 1 500

#keep box centered (no flying icecubes)
fix CoM_center_1 all recenter INIT INIT INIT units box

thermo_style custom step temp press vol etotal
thermo 1000

# run equilibriation
print "Running NPT equilibriation"
run %d

unfix ensemble_npt
unfix CoM_center_1
unfix water_shake_1

# write_dump all xyz npt_equil.xyz modify element Li Cl O H

##################
### Now do NVT ###
##################

fix ensemble_nvt all nvt temp 300 300 200
fix CoM_center_2 all recenter INIT INIT INIT units box
fix water_shake_2 water shake 1e-9 200 0 b 1 a 1

thermo_style custom step temp press vol etotal
thermo 1000

# run equilibriation
print "Running NVT equilibriation"
run %d

unfix ensemble_nvt
unfix CoM_center_2
unfix water_shake_2

# write_dump all xyz nvt_equil.xyz modify element Li Cl O H

##################
### Now do NVE ###
##################

# Fixes
fix ensemble_nve water nve
#fix CoM_center_3 all recenter INIT INIT INIT units box
#fix water_shake_3 water shake 1e-9 200 0 b 1 a 1

# Thermo
thermo_style custom step temp press vol etotal
thermo 1000

# Dumps
dump the_big_dump ion custom 5 explicit_run.lammpstrj id type x y z vx vy vz fx fy fz
# dump compress_traj all dcd 50 traj.dcd

# Restart files every 20ps
# restart 20000 system_1.restart system_2.restart

# run production
print "Running NVE production"
run %d

#write_dump all xyz final.xyz modify element Li Cl O H
"""%(time_step,npt_steps,nvt_steps,nve_steps)

lmp.commands_string(block)