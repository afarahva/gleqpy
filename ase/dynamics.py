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
                 int_type = 0, indices=None, temperature_K=None, 
                 temperature=None,  fixcm=False, trajectory=None, logfile=None, 
                 loginterval=1, append_trajectory=False, communicator=world):
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
            
        int_type: int
            Integrator algorithm to use for GLE. Default is 0 which uses
            Verlet scheme. 
            
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
        
        # Convert and Assign GLE Matrices
        if Amat_units == "ase":
            self.set_Amat(Amat, 1.0)
        elif Amat_units == "fs":
            self.set_Amat(Amat, units.fs)
        elif Amat_units == "ps":
            self.set_Amat(Amat, units.fs*1e3)
        else:
            raise ValueError(" 'Amat_units' must either be ase, ps, or fs")
            
        self.set_Bmat(None,1.0)
    
        # Choose integrator algorithm
        self.integrator = self.Verlet

        # Assign MPI communicator
        if communicator is None:
            communicator = DummyMPI()
        self.communicator = communicator
        
        # Initialize noise array
        self.sample_noise()
        
        pass


    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update({'temperature_K': self.temp / units.kB,
                  'Amat': self.Amat, 'Bmat': self.Bmat,
                  'fixcm': self.fix_com})
        return d

    def set_temperature(self, temperature_K):
        self.temp = units.kB * self._process_temperature(None, temperature_K, 
                                                         'eV')
        pass

    def set_Amat(self, Amat, unit_conv):
        """
        Sets GLE friction matrix
        """
        #convert units
        Amat = Amat/unit_conv
        
        self.naux = np.size(Amat, axis=0) - 1
        
        # Break apart A (friction) matrix
        self.Aps = Amat[0:1,1:]
        self.Asp = Amat[1:,0:1]
        self.As  = Amat[1:,1:]

        # Make auxiliary variable array
        self.s = np.zeros((self.ntherm,self.naux,3),dtype=np.float64)
        pass
    
    def set_Bmat(self, Bs, unit_conv):
        """
        Sets GLE B matrix according according to input to fluctuation-dissip
        theorem.
        """
        # Break apart B (Wiener) matrix
        if Bs is None:
            try:
                self.Bs = np.linalg.cholesky(self.temp * (self.As + self.As.T))
            except:
                self.Bs = np.sqrt(self.temp * (self.As + self.As.T))
        else:
            self.Bs = Bs

        #convert units
        self.Bs = self.Bs/unit_conv
        pass

    def sample_noise(self):
        """
        Sample noise vector
        """
        self.noise = normal(loc=0.0, scale=1.0, size=(self.ntherm,self.naux,3) )
        self.communicator.broadcast(self.noise, 0)
        pass
    
    def move_aux(self,p,dt):
        """
        Move auxiliary variables forward in time by dt
        """
        s_self = -np.einsum("ij,njd->nid", self.As, self.s)
    
        s_sys  = -np.einsum("if,nd->nid", self.Asp, p)
        
        s_ran = np.einsum("ij,njd->nid",self.Bs, self.noise) * self.sqrtmass
        
        self.s = self.s + (dt * s_self) + (dt * s_sys) + \
            (np.sqrt(dt) * s_ran)
        pass
    
    def Verlet(self, forces=None):
        """
        Type-1 velocity verlet algorithm. Auxiliary variables are moved with 
        system positions
        """
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
        self.sample_noise()
        self.move_aux(p[self.indices],self.dt)
        
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
        
    
    def step(self, forces=None):
        forces = self.integrator()
        return forces


class GLD_Aniso(MolecularDynamics):
    """Anisotropic Generalized Langevin (NVT) molecular dynamics."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 4

    def __init__(self, atoms, timestep, Amat_list, Amat_units = "ase",
                 indices=None, temperature_K=None, temperature=None, 
                 fixcm=False, trajectory=None, logfile=None, loginterval=1, 
                 append_trajectory=False, communicator=world):
        """
        Parameters:

        atoms: Atoms object
            The list of atoms.

        timestep: float
            The time step in ASE time units.
            
        Amat_list: numpy array.
            List of generalized Langevin friction matrix.
            
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
        
        # Assign which atoms are thermostatted by GLE
        if indices is None:
            self.indices = np.arange(self.nsys)
        else:
            self.indices = indices
            
        # number of GLE thermostated atoms
        self.ntherm = len(self.indices)
        
        # sqrt(mass) for GLE timestep
        self.sqrtmass = np.sqrt( self.masses.copy()[self.indices] )
    
        # Convert and Assign GLE Matrices
        if Amat_units == "ase":
            self.set_Amat(Amat_list, 1.0)
        elif Amat_units == "fs":
            self.set_Amat(Amat_list, units.fs)
        elif Amat_units == "ps":
            self.set_Amat(Amat_list, units.fs*1e3)
        else:
            raise ValueError(" 'Amat_units' must either be ase, ps, or fs")

        self.set_Bmat(None, 1.0)
        
        # Choose integrator algorith,
        self.integrator= self.Verlet
        
        # Assign MPI communicator
        if communicator is None:
            communicator = DummyMPI()
        self.communicator = communicator
        
        # Initialize noise array
        self.sample_noise()
        pass

    def todict(self):
        """
        Returns properties of integrator as dictionary
        """
        d = MolecularDynamics.todict(self)
        d.update({'temperature_K': self.temp / units.kB,
                  'fixcm': self.fix_com})
        return d

    def set_temperature(self, temperature_K):
        """
        Sets temperature of integrator
        Input in units of Kelvin. 
        """
        self.temp = units.kB * self._process_temperature(None, temperature_K, 
                                                         'eV')
        pass

    def set_Amat(self, Amat_list, unit_conv):
        """
        Sets GLE matrix for each direction.
        """
        #Convert units
        Amat_x, Amat_y, Amat_z = Amat_list
        Amat_x, Amat_y, Amat_z = Amat_x/unit_conv, Amat_y/unit_conv, Amat_z/unit_conv
        self.naux = np.size(Amat_x, axis=0) - 1

        # Break apart A_x (friction) matrix
        self.Aps_x = Amat_x[0:1,1:]
        self.Asp_x = Amat_x[1:,0:1]
        self.As_x  = Amat_x[1:,1:]
        
        # Break apart A_y (friction) matrix
        self.Aps_y = Amat_y[0:1,1:]
        self.Asp_y = Amat_y[1:,0:1]
        self.As_y  = Amat_y[1:,1:]
        
        # Break apart A_z (friction) matrix
        self.Aps_z = Amat_z[0:1,1:]
        self.Asp_z = Amat_z[1:,0:1]
        self.As_z  = Amat_z[1:,1:]
    
        # set auxiliary variable
        self.s = np.zeros((self.ntherm,self.naux,3),dtype=np.float64)
        pass
        
    def set_Bmat(self, Bmat_list, unit_conv):
        """
        Sets GLE B matrix according according to input to fluctuation-dissip
        theorem.
        """
        
        #If no argument is specified default to using cholesky decomposition
        if Bmat_list is None:
            try:
                self.Bs_x = np.linalg.cholesky(self.temp * (self.As_x + self.As_x.T))
                self.Bs_y = np.linalg.cholesky(self.temp * (self.As_y + self.As_y.T))
                self.Bs_z = np.linalg.cholesky(self.temp * (self.As_z + self.As_z.T))
            except:
                self.Bs_x = np.sqrt(self.temp * (self.As_x + self.As_x.T))
                self.Bs_y = np.sqrt(self.temp * (self.As_y + self.As_y.T))
                self.Bs_z = np.sqrt(self.temp * (self.As_z + self.As_z.T))
        else:
            self.Bs_x, self.Bs_y, self.Bs_z = Bmat_list
            
        #convert units
        self.Bs_x, self.Bs_y, self.Bs_z = self.Bs_x/unit_conv, self.Bs_y/unit_conv, self.Bs_z/unit_conv
        pass

    def sample_noise(self):
        """
        Sample noise vector
        """
        self.noise = normal(loc=0.0, scale=1.0, size=(self.ntherm,self.naux,3) )
        self.communicator.broadcast(self.noise, 0)
        pass
        
    
    def move_aux(self, p, dt):
        """
        Move auxiliary variables forward in time by dt
        """
        
        # Move X-auxiliary variables
        s_self = -np.einsum("ij,nj->ni", self.As_x, self.s[:,:,0])
        s_sys  = -np.einsum("if,n->ni", self.Asp_x, p[:,0])
        s_ran  = np.einsum("ij,nj->ni",self.Bs_x, self.noise[:,:,0]) * self.sqrtmass
        self.s[:,:,0] = self.s[:,:,0] + (dt * s_self) + (dt * s_sys) + (np.sqrt(dt) * s_ran)
        
        # Move Y-auxiliary variables
        s_self = -np.einsum("ij,nj->ni", self.As_y, self.s[:,:,1])
        s_sys  = -np.einsum("if,n->ni", self.Asp_y, p[:,1])
        s_ran  = np.einsum("ij,nj->ni",self.Bs_y, self.noise[:,:,1]) * self.sqrtmass
        self.s[:,:,1] = self.s[:,:,1] + (dt * s_self) + (dt * s_sys) + (np.sqrt(dt) * s_ran)
        
        # Move Z-auxiliary variables
        s_self = -np.einsum("ij,nj->ni", self.As_z, self.s[:,:,2])
        s_sys  = -np.einsum("if,n->ni", self.Asp_z, p[:,2])
        s_ran  = np.einsum("ij,nj->ni",self.Bs_z, self.noise[:,:,2]) * self.sqrtmass
        self.s[:,:,2] = self.s[:,:,2] + (dt * s_self) + (dt * s_sys) + (np.sqrt(dt) * s_ran)
        
        pass
    
    def Verlet(self,forces=None):
        """
        Type-1 velocity verlet algorithm. Auxiliary variables are moved with 
        system positions
        """
        atoms = self.atoms

        if forces is None:
            forces = atoms.get_forces(md=True)

        # move momenta half step
        p = atoms.get_momenta()
        p = p + 0.5 * self.dt * forces
        
        p[self.indices,0] = p[self.indices,0] \
            - 0.5 * self.dt * np.einsum("fj,nj->n", self.Aps_x, self.s[:,:,0])
        p[self.indices,1] = p[self.indices,1] \
            - 0.5 * self.dt * np.einsum("fj,nj->n", self.Aps_y, self.s[:,:,1])
        p[self.indices,2] = p[self.indices,2] \
            - 0.5 * self.dt * np.einsum("fj,nj->n", self.Aps_z, self.s[:,:,2])
        
        # move positions whole step
        r = atoms.get_positions()   
        if self.fix_com:
            old_com = atoms.get_center_of_mass()
        atoms.set_positions(r + self.dt * p / self.masses)
        if self.fix_com:
            atoms.set_center_of_mass(old_com)
        
        # move auxiliary variables full-step
        self.sample_noise()
        self.move_aux(p[self.indices],self.dt)
        
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
            
        # move momenta half step
        p = atoms.get_momenta()
        p = p + 0.5 * self.dt * forces
        
        p[self.indices,0] = p[self.indices,0] \
            - 0.5 * self.dt * np.einsum("fj,nj->n", self.Aps_x, self.s[:,:,0])
        p[self.indices,1] = p[self.indices,1] \
            - 0.5 * self.dt * np.einsum("fj,nj->n", self.Aps_y, self.s[:,:,1])
        p[self.indices,2] = p[self.indices,2] \
            - 0.5 * self.dt * np.einsum("fj,nj->n", self.Aps_z, self.s[:,:,2])
            
        atoms.set_momenta(p)
        return forces
    
    def step(self, forces=None):
        forces = self.integrator()
        return forces
    
class Langevin_Custom(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 5

    def __init__(self, atoms, timestep, indices=None, temperature=None, friction=None,
                 fixcm=True, *, temperature_K=None, trajectory=None,
                 logfile=None, loginterval=1, communicator=world,
                 rng=None, append_trajectory=False):
        """
        Parameters:

        atoms: Atoms object
            The list of atoms.

        timestep: float
            The time step in ASE time units.
            
        indices: list (optional)
            indices of atoms in contact with bath. Use *None* 
            to apply bath to all atoms.

        temperature: float (deprecated)
            The desired temperature, in electron volt.

        temperature_K: float
            The desired temperature, in Kelvin.

        friction: float
            A friction coefficient, typically 1e-4 to 1e-2.

        fixcm: bool (optional)
            If True, the position and momentum of the center of mass is
            kept unperturbed.  Default: True.

        rng: RNG object (optional)
            Random number generator, by default numpy.random.  Must have a
            standard_normal method matching the signature of
            numpy.random.standard_normal.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str (optional)
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* (the default) for no
            trajectory.

        communicator: MPI communicator (optional)
            Communicator used to distribute random numbers to all tasks.
            Default: ase.parallel.world. Set to None to disable communication.

        append_trajectory: bool (optional)
            Defaults to False, which causes the trajectory file to be
            overwritten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.

        The temperature and friction are normally scalars, but in principle one
        quantity per atom could be specified by giving an array.

        RATTLE constraints can be used with these propagators, see:
        E. V.-Eijnden, and G. Ciccotti, Chem. Phys. Lett. 429, 310 (2006)

        The propagator is Equation 23 (Eq. 39 if RATTLE constraints are used)
        of the above reference.  That reference also contains another
        propagator in Eq. 21/34; but that propagator is not quasi-symplectic
        and gives a systematic offset in the temperature at large time steps.
        """
        if friction is None:
            raise TypeError("Missing 'friction' argument.")
        self.fr = friction
        self.temp = units.kB * self._process_temperature(temperature,
                                                         temperature_K, 'eV')
        self.fix_com = fixcm
        if communicator is None:
            communicator = DummyMPI()
        self.communicator = communicator
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng
        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval,
                                   append_trajectory=append_trajectory)
        self.updatevars()
        
        # number of atoms in system
        self.nsys = len(atoms) 
        
        # Assign which atoms to interact with GLE thermostat
        if indices is None:
            self.indices = np.arange(self.nsys)
        else:
            self.indices = indices

    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update({'temperature_K': self.temp / units.kB,
                  'friction': self.fr,
                  'fixcm': self.fix_com})
        return d

    def set_temperature(self, temperature=None, temperature_K=None):
        self.temp = units.kB * self._process_temperature(temperature,
                                                         temperature_K, 'eV')
        self.updatevars()

    def set_friction(self, friction):
        self.fr = friction
        self.updatevars()

    def set_timestep(self, timestep):
        self.dt = timestep
        self.updatevars()

    def updatevars(self):
        dt = self.dt
        T = self.temp
        fr = self.fr
        masses = self.masses
        sigma = np.sqrt(2 * T * fr / masses)

        self.c1 = dt / 2. - dt * dt * fr / 8.
        self.c2 = dt * fr / 2 - dt * dt * fr * fr / 8.
        self.c3 = np.sqrt(dt) * sigma / 2. - dt**1.5 * fr * sigma / 8.
        self.c5 = dt**1.5 * sigma / (2 * np.sqrt(3))
        self.c4 = fr / 2. * self.c5

    def step(self, forces=None):
        atoms = self.atoms
        natoms = len(atoms)

        if forces is None:
            forces = atoms.get_forces(md=True)

        # This velocity as well as rnd_pos, rnd_mom and a few other variables are stored
        # as attributes, so Asap can do its magic when atoms migrate between
        # processors.
        self.v = atoms.get_velocities()

        xi = self.rng.standard_normal(size=(natoms, 3))
        eta = self.rng.standard_normal(size=(natoms, 3))

        # When holonomic constraints for rigid linear triatomic molecules are
        # present, ask the constraints to redistribute xi and eta within each
        # triple defined in the constraints. This is needed to achieve the
        # correct target temperature.
        for constraint in self.atoms.constraints:
            if hasattr(constraint, 'redistribute_forces_md'):
                constraint.redistribute_forces_md(atoms, xi, rand=True)
                constraint.redistribute_forces_md(atoms, eta, rand=True)

        self.communicator.broadcast(xi, 0)
        self.communicator.broadcast(eta, 0)

        # To keep the center of mass stationary, we have to calculate the random
        # perturbations to the positions and the momenta, and make sure that they
        # sum to zero.
        self.rnd_pos = self.c5 * eta
        self.rnd_vel = self.c3 * xi - self.c4 * eta
        if self.fix_com:
            self.rnd_pos -= self.rnd_pos.sum(axis=0) / natoms
            self.rnd_vel -= (self.rnd_vel * self.masses).sum(axis=0) / (self.masses * natoms)

        # First halfstep in the velocity.
        self.v += (self.c1 * forces / self.masses - self.c2 * self.v +
                   self.rnd_vel)

        # Full step in positions
        x = atoms.get_positions()

        # Step: x^n -> x^(n+1) - this applies constraints if any.
        atoms.set_positions(x + self.dt * self.v + self.rnd_pos)

        # recalc velocities after RATTLE constraints are applied
        self.v = (self.atoms.get_positions() - x - self.rnd_pos) / self.dt
        forces = atoms.get_forces(md=True)

        # Update the velocities
        self.v += (self.c1 * forces / self.masses - self.c2 * self.v +
                   self.rnd_vel)

        # Second part of RATTLE taken care of here
        atoms.set_momenta(self.v * self.masses)

        return forces
