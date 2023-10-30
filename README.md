GLEPy
=====

*generalized Langevin equation with python*

This repo contains tools for the simulation of the generalized Langevin equation (GLE) and the 
calculation of memory/friction kernels. 

The GLE is a non-Markovian counterpart to the Langevin equation,

$$ \mathbf{p} = -\frac{d W}{d \mathbf{x}}(t) - \int_0^t \mathbf{K}(t-\tau) \mathbf{p}(\tau) d\tau + \mathbf{R}(t) $$

where $\mathbf{p}$ and $\mathbf{x}$ are the momenta and positions of your system, $W$ is 
the potential of mean force, $\mathbf{K}$ is the memory kernel (generally a tensor), 
and $\mathbf{R}$ is a correlated stochastic process. 

<p align="center">
<img src="https://github.com/afarahva/glepy/examples/1D/memory.png" width="500">
</p>

The figure above compares memory kernels, one that was used as an input for a 
simulation and others that were extracted by analyzing the data from that simulation.

Pedagogical examples of how to set up, run, and analyze GLE simulations are provided in the 
*examples* directory.


Getting Started
---------------

The easiest way to install glepy is by using `pip`.

The memory extraction scripts only require numpy and scipy.

The ase module requires the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/index.html)
to be installed.

The examples directory contains discusses how to use [LAMMPS](https://www.lammps.org/) to run GLE simulations.
 
Submodules
----------

ase - GLE integrators and helpful forcefields for Atomic Simulation Environment 

examples - Helpful examples.

memory - Functions for calculating memory kernels, and a database for memory kernels 
calculated from prior simulations. 

md - Python based MD code, useful for testing purposes and building toy simulations. 


Citing glepy
------------
If you use glepy, please cite: