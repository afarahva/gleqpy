 # -*- coding: utf-8 -*-
"""
GLEPy
============

submodule: LAMMPS
file: dynamics.py
author: Ardavan Farahvash (MIT)

description: 
GLE thermostat implementation of ASE python package. 
"""
import numpy as np
from numpy.random import normal

from ase.md.md import MolecularDynamics
from ase.parallel import world, DummyMPI
from ase import units


class GLD(MolecularDynamics):
    """Generalized Langevin (constant N, V, T) molecular dynamics."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 4

    def __init__(self, atoms, timestep, Amat, Amat_units = "ase",
                 indices=None, temperature_K=None, temperature=None, 
                 fixcm=False, trajectory=None, logfile=None, loginterval=1, 
                 append_trajectory=False, communicator=world):
        """
        Parameters:

        atoms: Atoms object
            The list of atoms.

        timestep: float
            The time step in ASE time units.
            
        Amat: numpy array.
            Generalized Langevin friction matrix.
            
        Amat_units: numpy array.
            Units for friction matrix. Default is 1/(Ase time units).
            Other choices are "ps" for 1/ps or "fs" for 1/fs
            
        indices: list (optional)
            indices of atoms in contact with bath. Use *None* 
            to apply bath to all atoms.

        temperature: float (deprecated)
            The desired temperature, in electron volt.

        temperature_K: float
            The desired temperature, in Kelvin.

        fixcm: bool (optional)
            If True, the position and momentum of the center of mass is
            kept unperturbed.  Default: False.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str (optional)
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* (the default) for no
            trajectory.

        append_trajectory: bool (optional)
            Defaults to False, which causes the trajectory file to be
            overwritten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.
            
        communicator: MPI communicator (optional)
                    Communicator used to distribute random numbers to  tasks.
                    Default: ase.parallel.world. Set to None to disable.
        """
        
        
        # Assign and convert temperature    
        self.temp = units.kB * self._process_temperature(temperature, 
                                                         temperature_K, 'eV')
        
        # Assign other class variables
        self.fix_com = fixcm

        MolecularDynamics.__init__(self, atoms, timestep, trajectory, logfile,
                                   loginterval,
                                   append_trajectory=append_trajectory)
        
        # Masses
        self.masses = atoms.get_masses()[:, None]
        
        # number of atoms in system
        self.nsys = len(atoms) 
        
        # Assign which atoms to interact with GLE thermostat
        if indices is None:
            self.indices = np.arange(self.nsys)
        else:
            self.indices = indices
            
        # number of GLE thermostated atoms
        self.ntherm = len(self.indices)
        
        # sqrt(mass) for GLE timestep
        self.sqrtmass = np.sqrt( self.masses.copy()[self.indices,None] )
        
        # Assign MPI communicator
        if communicator is None:
            communicator = DummyMPI()
        self.communicator = communicator
        
        # Convert and Assign Bath Parameters
        if Amat_units == "ase":
            self.set_bathparms(Amat, None)
        elif Amat_units == "fs":
            self.set_bathparms(Amat/units.fs, None)
        elif Amat_units == "ps":
            self.set_bathparms(Amat/units.fs/1e3, None)
        else:
            raise ValueError(" 'Amat_units' must either be ase, ps, or fs")


    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update({'temperature_K': self.temp / units.kB,
                  'Amat': self.Amat, 'Bmat': self.Bmat,
                  'fixcm': self.fix_com})
        return d

    def set_temperature(self, temperature_K):
        self.temp = units.kB * self._process_temperature(None, temperature_K, 
                                                         'eV')

    def set_bathparms(self, Amat, Bs):
    
        # Break apart A (friction) matrix
        self.naux = np.size(Amat, axis=0) - 1
        self.Aps = Amat[0:1,1:]
        self.Asp = Amat[1:,0:1]
        self.As  = Amat[1:,1:]
        
        # Break apart B (Wiener) matrix
        if np.all(Bs) == None:
            try:
                self.Bs = np.linalg.cholesky(self.temp * (self.As + self.As.T))
            except:
                self.Bs = np.sqrt(self.temp * (self.As + self.As.T))
        else:
            self.Bs = Bs

        self.s = np.zeros((self.ntherm,self.naux,3),dtype=np.float64)
    
    def step_aux(self,p):
        s_self = -np.einsum("ij,njd->nid", self.As, self.s)
    
        s_sys  = -np.einsum("if,nd->nid", self.Asp, p)
        
        noise = normal(loc=0.0, scale=1.0, size=(self.ntherm,self.naux,3) )
        self.communicator.broadcast(noise, 0)
        s_ran = np.einsum("ij,njd->nid",self.Bs,noise) * self.sqrtmass
        
        self.s = self.s + (self.dt * s_self) + (self.dt * s_sys) + \
            (np.sqrt(self.dt) * s_ran)

    
    def step(self, forces=None):

        atoms = self.atoms

        if forces is None:
            forces = atoms.get_forces(md=True)

        # move momenta half step
        p = atoms.get_momenta()
        p = p + 0.5 * self.dt * forces
        p[self.indices] = p[self.indices] \
            - 0.5 * self.dt * np.einsum("fj,njd->nd", self.Aps, self.s)
        
        # Move positions whole step
        r = atoms.get_positions()   
        if self.fix_com:
            old_com = atoms.get_center_of_mass()
        atoms.set_positions(r + self.dt * p / self.masses)
        if self.fix_com:
            atoms.set_center_of_mass(old_com)
        
        # Move auxiliary variables full-step
        self.step_aux(p[self.indices])
        
        # if we have constraints then this will do the first part of the
        # RATTLE algorithm:
            
        if atoms.constraints:
            p = (atoms.get_positions() - r) * self.masses / self.dt

        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.
        atoms.set_momenta(p, apply_constraint=False)

        forces = atoms.get_forces(md=True)

        # Second part of RATTLE will be done here:
        # move momenta half step
        p = atoms.get_momenta()
        p = p + 0.5 * self.dt * forces
        p[self.indices] = p[self.indices] \
            - 0.5 * self.dt * np.einsum("fj,njd->nd", self.Aps, self.s)
            
        atoms.set_momenta(p)
        return forces
