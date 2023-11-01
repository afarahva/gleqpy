GLEPy
=====

*Generalized Langevin Equation with Python*

This repo contains tools for the simulation of the generalized Langevin equation (GLE) and 
the calculation of memory/friction kernels. 

The GLE is a non-Markovian counterpart to the Langevin equation,

$$ \mathbf{p} = -\frac{d W}{d \mathbf{x}}(t) - \int_0^t \mathbf{K}(t-\tau) \mathbf{p}(\tau) d\tau + \mathbf{R}(t) $$

where $\mathbf{p}$ and $\mathbf{x}$ are the momenta and positions of your system, $W$ is 
the potential of mean force, $\mathbf{K}$ is the memory kernel (generally a tensor), 
and $\mathbf{R}$ is a correlated stochastic process. 

<p align="center">
<img src="https://raw.githubusercontent.com/afarahva/glepy/main/examples/1D/memory.png" width="500">
</p>

The figure above compares memory kernels, one that was used as an input for a 
simulation and others that were extracted by analyzing the data from that simulation.

Pedagogical examples of how to set up, run, and analyze GLE simulations are provided in the 
**examples** directory. Examples include toy simulations, GLE for solid dynamics with ASE, 
and GLE for solution phase dynamics with LAMMPS.


Getting Started
---------------

The easiest way to install glepy is by using `pip`.

The memory analysis tools only require numpy and scipy.

The ase module requires the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/index.html)
to be installed.

The examples directory contains discusses how to use [LAMMPS](https://www.lammps.org/) to run GLE simulations.

New to the GLE
--------------

New to the GLE? Want a short tutorial on the motivation and formalism? 
The Mathematica notebook in the **examples** directory contains a helpful guide. 

 
Submodules
----------

**ase** - GLE integrators and helpful forcefields for Atomic Simulation Environment. 

**examples** - Helpful examples.

**memory** - Functions for calculating memory kernels, and a database for memory kernels 
calculated from prior simulations. 

**md** - Python based MD code, useful for testing purposes and building toy simulations.


Citing glepy
------------
If you use glepy, please cite: 
Farahvash A, Agrawal M, Peterson AA, Willard AP. Modeling Surface Vibrations and Their Role in Molecular Adsorption: A Generalized Langevin Approach. J Chem Theory Comput. 2023 Sep 26;19(18):6452-6460. doi: [10.1021/acs.jctc.3c00473](10.1021/acs.jctc.3c00473). Epub 2023 Sep 8.
