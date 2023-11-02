 # -*- coding: utf-8 -*-
"""
GLEqPy
============

submodule: md
file: forcefield.py
author: Ardavan Farahvash

description: 
Simple forcefield classes to use with python GLE simulation integrators. 
"""

import numpy as np
from itertools import combinations

class ff_zero(object):
    """
    Free particle forcefield object.  
    """
    def __init__(self,natom,ndim):
        self.natom = natom
        self.ndim = ndim
        pass
    def calc_pot(self,x):
        return np.zeros((self.natom,self.ndim),dtype=np.float64)
    def calc_frc(self,x):
        return np.zeros((self.natom,self.ndim),dtype=np.float64) 
    
class ff_harm(object):
    """
    Isotropic harmonic oscillator:
    U(r) = 1/2 k (x - x_0)^2
    """
    def __init__(self,natom,ndim,frck,xeq,box_l=None):
        self.natom = natom
        self.ndim = ndim
        self.frck = frck
        self.xeq = xeq
        self.box_l = box_l
        pass
    
    def calc_pot(self,x):
        """
        Calculate Potential energy
        """
        diff = (x - self.xeq)

        if self.box_l is not None:
            diff = np.where(diff > self.box_l/2, diff-self.box_l, diff)
            diff = np.where(diff < -self.box_l/2, diff+self.box_l, diff)
        
        pot =  np.sum( 0.5 * self.frck * diff**2  )
        return pot
    
    def calc_frc(self,x):
        """
        Calculate Force
        """
        diff = (x - self.xeq)
        
        if self.box_l is not None:
            diff = np.where(diff > self.box_l/2, diff-self.box_l, diff)
            diff = np.where(diff < -self.box_l/2, diff+self.box_l, diff)
            
        frc =  - self.frck * diff
        return frc

class ff_anisoharm(object):
    """
    Anisotropic harmonic oscillator:
    U(r) = 1/2 (x - x_0).T @ k @ (x - x_0)
    """
    def __init__(self,natom,ndim,frck,xeq,box_l=None):
        self.natom = natom
        self.ndim = ndim
        self.frck = frck
        self.xeq = xeq
        self.box_l = box_l
        pass
    
    def calc_pot(self,x):
        """
        Calculate Potential energy
        """
        diff = (x - self.xeq)

        if self.box_l is not None:
            diff = np.where(diff > self.box_l/2, diff-self.box_l, diff)
            diff = np.where(diff < -self.box_l/2, diff+self.box_l, diff)
        
        pot =  0.5 * np.einsum("di,ij,dj->",diff,self.frck,diff)
        return pot
    
    def calc_frc(self,x):
        """
        Calculate Force
        """
        diff = (x - self.xeq)
        
        if self.box_l is not None:
            diff = np.where(diff > self.box_l/2, diff-self.box_l, diff)
            diff = np.where(diff < -self.box_l/2, diff+self.box_l, diff)
            
        frc =  - np.einsum("ij,dj->di",self.frck,diff)
        return frc

class ff_morseZ(object):
    """
    Morse Potential - Z-Direction Only
    U(x) = D  ( 1 - e^(-a (z - z_0)) )^2
    """
    def __init__(self,natom, m, D, a, z_0):
        self.natom = natom
        self.ndim  = 3
        
        self.m = m        
        self.D = D        
        self.a = a
        self.z_0 = z_0
        pass
        
    def calc_pot(self,x):
        dist = x[:,2] - self.z_0[:,2]
        pot = self.D * ( 1 - np.exp(-self.a * dist) ) ** 2  
        return pot
    
    def calc_frc(self,x):
        displ = np.zeros((self.natom,self.ndim),dtype=np.float64)
        displ[:,2] = x[:,2] - self.x_0[:,2]
        dist = displ[:,2]
        frc = - 2 * self.D * self.a * \
            ( np.exp(-self.a * dist) - np.exp(-2*self.a * dist) ) * displ/dist
        return frc
    
class ff_morse1(object):
    """
    Morse Potential - Type 1
    U(x) = D  ( 1 - e^(-a (x - x_0)) )^2
    """
    def __init__(self,natom,ndim, m, D, a, x_0):
        self.natom = natom
        self.ndim  = ndim
        
        self.m = m        
        self.D = D        
        self.a = a
        self.x_0 = x_0
        pass
        
    def calc_pot(self,x):
        dist = np.linalg.norm(x - self.x_0, axis=1)
        pot = self.D * ( 1 - np.exp(-self.a * dist) ) ** 2  
        return pot
    
    def calc_frc(self,x):
        displ = x - self.x_0
        dist = np.linalg.norm(displ, axis=1)
        frc = - 2 * self.D * self.a * \
            ( np.exp(-self.a * dist) - np.exp(-2*self.a * dist) ) * displ/dist
        return frc
    
class ff_morse2(object):
    """
    Morse Potential - Type 2
    U(x) = D  ( e^(-2/a (x - x_0)) - e^(-1/a (x - x_0)) )
    """
    def __init__(self,natom,ndim, m, D, a, x_0):
        """
        Parameters
        ----------
        natom : Int.
            Number of atoms.
        ndim : Int.
            Number of dimensions.
        m : Numpy array.
            Mass of Particles.
        D : Numpy Array.
            Morse Potential Depth.
        a : Numpy Array.
            Morse Potential Width.
        x_0 : Numpy Array.
            Location of Potential Minimum.
        """
        self.natom = natom
        self.ndim  = ndim
        
        self.m = m        
        self.D = D        
        self.a = a
        self.x_0 = x_0
        pass
        
    def calc_pot(self,x):
        dist = np.linalg.norm(x - self.x_0, axis=1)
        pot = self.D * ( np.exp(-2/self.a*dist)  - 2 * np.exp(-1/self.a*dist )  )
        return pot
    
    def calc_frc(self,x):
        displ = x - self.x_0
        dist = np.linalg.norm(displ, axis=1)
        frc = -2*self.D/self.a * \
            ( np.exp(-1/self.a*dist) - np.exp(-2/self.a*dist) ) * displ/dist
        return frc
        
class ff_quartic(object):
    """
    1D Quartic Potential Energy
    U(x) = C4*x^4 + C3*x^3 + C2*x^2 +C1*x + C0
    """
    def __init__(self,natom,height,C4,C3,C2,C1,C0):
        self.height = height
        self.C4 = C4
        self.C3 = C3
        self.C3 = C3
        self.C2 = C2
        self.C1 = C1
        self.C0 = C0

        self.natom = natom
        self.ndim = 1
        pass
    
    def calc_freqs(self,x):
        """
        Calculates Frequencies of oscillations near local minima of U(x)
        """
        pot = self.calc_pot(x)
        sort = np.argsort(pot)
        xmin1 = x[sort[0]]
        xmin2 = x[sort[1]]
        
        k1 = self.height * ( (12*self.C4)*(xmin1**2) + (6*self.C3)**2 * xmin1 + (2*self.C2) )
        k2 = self.height * ( (12*self.C4)*(xmin2**2) + (6*self.C3)**2 * xmin2 + (2*self.C2) )
        return xmin1, xmin2, k1, k2
        
    def calc_pot(self,x):
        poly = self.C4 * x**4 + self.C3 * x**3 + self.C2 * x**2 + self.C1 * x + self.C0
        pot = np.sum(self.height * poly, axis=1)
        return pot
    
    def calc_frc(self,x):
        poly = 4 * self.C4 * x**3 + 3 * self.C3 * x**2 + 2 * self.C2 * x + self.C1
        frc = -self.height * poly
        return frc
    
class ff_lennard_jones:
    """ ForceField object for Lennard-Jones potential """
    
    def __init__(self, eps, sigma, cutoff=-1):
        # Basic Forcefield Parameters
        self.eps = eps
        self.sigma = sigma
        self.cutoff = cutoff
        natom = np.size(eps,axis=0)
        self.natom = natom

        # Pairwise atomic parameters
        if natom > 1:
            self.npairs = int(natom * (natom-1)/2)
            pindx = list(combinations(range(natom), r=2))
            self.pindx_r = np.array(pindx)
            
            # Create eps_ij and sigma_ij as collapsed 1D arrays
            eps_ij = self.eps[:,None,0] * self.eps[None,:,0] 
            self.eps_ij = np.sqrt(eps_ij[self.pindx_r[:,0], self.pindx_r[:,1]])
            
            sigma_ij = self.sigma[:,None,0] + self.sigma[None,:,0] 
            self.sigma_ij = 0.5* sigma_ij[self.pindx_r[:,0],self.pindx_r[:,1]]

        # If there is only one atom in the system
        else:
            self.npairs = 0
            self.pindx_r = None
            self.sigma_ij = 0
            self.eps_ij = 0
            
    def calc_pot(self,pos,box_l):
        """
        Calculates potential energy betweeen all pairs of particles.
        Returns total potential energy.

        Parameters
        ----------
        pos : Numpy Array. (natom,3)
            Positions of atoms.
        box_l : Int.
            PBC box length.

        Returns
        -------
        pos : float.
            Total potential energy.
        """
        
        if self.npairs == 0:
            return 0
                
        # Pairwise Difference Vector
        diff = pos[:,None,:] - pos
        diff = diff[self.pindx_r[:,0],self.pindx_r[:,1],:]
        
        #Adjust for PBC, Minimum Image Convention
        diff = np.where(diff > box_l/2, diff-box_l, diff)
        diff = np.where(diff < -box_l/2, diff+box_l, diff)
            
        #Pairwise Distances
        dist = np.linalg.norm(diff,axis=1)
        
        #Pairwise Potential
        self.pairwise_pot = 4*self.eps_ij*( self.sigma_ij**12/(dist**12) - (self.sigma_ij**6)/(dist**6))
        
        #Remove entries beyond cutoff
        if self.cutoff > 0:
            self.pairwise_pot = np.where(dist > self.cutoff, 0.0, self.pairwise_pot)
        
        #Total Potential
        pot = np.sum(self.pairwise_pot)
        
        return pot 

    def calc_frc(self,pos,box_l):
        """
        Calculates forces on all atoms.

        Parameters
        ----------
        pos : Numpy Array. (natom,3)
            Positions of all atoms.
        box_l : Int.
            PBC box length.

        Returns
        -------
        frc : Numpy Array. (natom,3)
            Forces on all atoms.
        """
        
        
        ndim = np.size(pos,axis=1)
        pfrc_sq = np.zeros((self.natom,self.natom,ndim))
                
        if self.npairs == 0:
            return np.zeros((self.natom,ndim))
        
        # Pairwise Difference Vector
        diff = pos[:,None,:] - pos
        diff = diff[self.pindx_r[:,0],self.pindx_r[:,1],:]
        
        #Adjust for PBC, Minimum Image Convention
        diff = np.where(diff > box_l/2, diff-box_l, diff)
        diff = np.where(diff < -box_l/2, diff+box_l, diff)
            
        # Pairwise Distances
        dist = np.linalg.norm(diff,axis=1)
        
        # Calculate Pairwise Forces
        coef = 48*self.eps_ij*( self.sigma_ij**12/(dist**14) - 0.5*(self.sigma_ij**6)/(dist**8))
        pfrc =  coef[:,None] * diff
        
        # Reshape and sum
        pfrc_sq[self.pindx_r[:,0],self.pindx_r[:,1],:] =  pfrc
        pfrc_sq = pfrc_sq - np.swapaxes(pfrc_sq,0,1)
        
        frc = np.sum(pfrc_sq,axis=1)
        
        return frc
  
# To - Do
class ff_atomic:
    """ ForceField object for LJ + Coulomb Potential """