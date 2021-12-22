 # -*- coding: utf-8 -*-
"""
GLE-Py
============

submodule: LAMMPS
file: dynamics.py
author: Ardavan Farahvash (MIT)

description: 
Generalized Langevin Dynamics implementation for LAMMPS
"""

import numpy as np
from lammps import lammps
lmp = lammps()

class gld(object):
    """ 
    Class for integrating Generalized Langevin Dynamics.
    """
    
    def __init__(self,system, forcefield, dt, temp, As, Avs, Asv, Bs = None,  
                 PBC = False, reporters = [], reportints = []):
        """
        Parameters
        ----------
        system : System Object.
        forcefield : Forcefield Object.
        nt : Int.
            Number of timesteps.
        dt : Float.
            Length of each timestep.d
        
        temp : Float.
            Temperature. (Units are Energy/k_b)
            
        As : Numpy Array.
            Ornstein-Uhlenbeck Drift Matrix
        Bs : Numpy Array.
            Ornstein-Uhlenbeck Random Multiplier Matrix
        Asv : Numpy Array.
            System -> Auxiliary projection Matrix
        Asv : Numpy Array.
            Auxiliary ->  System projection Matrix
            
        Optional
        ----------
        PBC : Bool. 
            Whether or not to use periodic boundary conditions, Default=False
        reporters : list of reporter objects
        reportints : list of reporter intervals
        """
        
        #Set General Integration Parameters
        self.system = system
        self.ff = forcefield
        self.dt = dt
        self.PBC = PBC
        self.reporters = reporters
        self.reportints = reportints
        
        # Set Force Array
        self.f_t = np.zeros((system.nsys, system.ndim),dtype=np.float64)
        
        #Set Bath Parameters
        self.set_bathparms( temp, As, Avs, Asv, Bs = Bs)
        
        #Set Bath State
        self.s_t = np.zeros((system.nsys, self.naux, system.ndim)
                          , dtype=np.float64)

        pass
            
    def set_bathparms(self, temp, As, Avs, Asv, Bs=None, k0=None):
        """
        Set Ornstein-Uhlenbeck Bath Parameters.
        
        K(t) = k0. Avs . exp(-As t) . Asv ~ scalar
        k0 ~ Bs Bs.T / (2 As + As.T) (when As and Bs commute)
        
        Parameters
        ----------
        As : Numpy Array.
            Ornstein-Uhlenbeck Drift Matrix
        Bs : Numpy Array.
            Ornstein-Uhlenbeck Random Multiplier Matrix
        Asv : Numpy Array.
            System -> Auxiliary projection Matrix
        Asv : Numpy Array.
            Auxiliary ->  System projection Matrix
        temp : Float.
            Temperature. (Units are Energy/k_b)
        k0 : Numpy Array. 
            Ornstein-Uhlenbeck stationary variance. Default set to identity. 
        """
        
        # Assign Parameters
        self.As = As
        self.Asv = Asv
        self.Avs = Avs
        self.kbT = kb * temp
        
        # Assign Random Forces following Fluctuation-Dissipation thm,
        if np.all(Bs) == None:
            print("Assigning B matrix via Cholesky Decomposition")
            self.Bs = np.linalg.cholesky(self.kbT * (self.As + self.As.T))
        
        # Ensure All dimensions Match
        self.naux = np.size(As,axis=0)
        
        assert self.naux == np.size(self.Bs,axis=0) == np.size(self.Bs,axis=1)
        assert self.naux == np.size(self.Avs,axis=1) == np.size(self.Asv,axis=0)
        assert 1 == np.size(self.Avs,axis=0) == np.size(self.Asv,axis=1)
            
        # If k0 is not assigned, assume it is identity or already folded into Asv
        if np.all(k0) != None:
            self.Asv = k0 @ self.Asv
                        
        pass
        
    def run(self, nt):
        """
        Run dynamics for nt steps.
        """

        #Initial Forces
        self.f_t = self.ff.calc_frc(self.system.pos)
        
        # Report Initial State
        for k in range(len(self.reportints)):
            self.reporters[k].save(self.system.pos,self.system.vel,
                                   self.f_t/self.system.m,self.f_t)
        for i in range(nt):
            x_t, v_t, a_t, f_t = self.step()
        
            # Report Simulation Data
            for k in range(len(self.reportints)):
                if i % self.reportints[k]  == 0:
                    self.reporters[k].save(x_t,v_t,a_t,f_t)
        pass     

    def step(self):
        """
        Single Step of Generalized Langevin dynamics integrator
        """
    
        # Number of atoms/dimensions
        nsys  = self.system.nsys
        ndim  = self.system.ndim
        naux  = self.np.size(self.Bs,axis=0)
        dt = self.dt
        
        # mass
        m = self.system.m
        
        # Create Position Array
        x_t = self.system.pos
        
        # Create Velocity Array
        v_t = self.system.vel
        
        #Move velocity half-step
        a_t = self.f_t/m - np.einsum("fj,njd->nd", self.Avs, self.s_t)
        v_t = v_t + dt/2.0 * a_t
        
        # Move Position full-step
        x_t = x_t + dt * v_t
        if self.PBC:
            x_t = PBCwrap(x_t, self.system.box_l)
            
        # Calculate Force at new position
        self.f_t = self.ff.calc_frc(x_t)
        
        # Move auxiliary variables full-step
        s_self = -np.einsum("ij,njd->nid", self.As, self.s_t)
        
        s_sys  = -np.einsum("if,nd->nid", self.Asv, self.v_t)
            
        noise = np.random.normal(loc=0.0, scale=1.0, size=(nsys,naux,ndim))
        s_ran = np.einsum("ij,njd->nid",self.Bs,noise) / np.sqrt(m[:,None])
            
        self.s_t = self.s_t + (dt * s_self) + (dt * s_sys) + (np.sqrt(dt) * s_ran)
    
        # Move velocity half-step
        a_t = self.f_t/m - np.einsum("fj,njd->nd", self.Avs, self.s_t)
        v_t = v_t + dt/2.0 * a_t
            
        # Update System Object
        self.system.pos = x_t.copy()
        self.system.vel = v_t.copy()
        
        # Function returns final state of auxiliary variables
        return x_t, v_t, a_t, self.f_t
