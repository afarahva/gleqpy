 # -*- coding: utf-8 -*-
"""
file: ion.py
author: Ardavan Farahvash (MIT)

description: 
Python tools from calculating pairwise ion potential of mean forces.
"""

from itertools import combinations
import numpy as np
from scipy.interpolate import interp1d

class pairwise_potentials(object):
    """
    Calculates various pairwise additive potential energies. By default outputs
    energies in kJ/mol
    """    

    def __init__(self,dist):
        """
        Initialize object by providing a set of distances between the atoms.

        Parameters
        ----------
        dist : Float or Numpy Array.
            Distances between atoms.
        """
        self.dist = dist
    
    def coulomb(self,q_1,q_2,eps=1,k_md = 138.935458):
        """
        Calculate Coulomb potential.

        Parameters
        ----------
        q_1 : Float.
            Charges on atom 1 
        q_2 : Float.
            Charges on atom 2
        eps : float, optional
            Dielectric Constant. The default is 1.
        k_md : float, optional
            Coulomb Constant. The default is 138.935458 kJ/mol/nm. See ~ http://manual.gromacs.org/documentation/2019/reference-manual/definitions.html

        Returns
        -------
        pot : Numpy Array.
             Potential Energy
        """
        pot = 1/eps * k_md * (q_1 * q_2)/(self.dist)
        return pot
    
    def lj(self,eps_1,sigma_1,eps_2,sigma_2):
        """
        Calculate Lennard-Jones potential.
        
        Parameters
        ----------
        eps_1 : Float.
            Epsilon param on atom 1 
        sigma_1 : Float.
            Sigma param on atom 1 
        eps_2 : Float.
            Epsilon param on atom 1 
        sigma_2 : Float.
            Sigma param on atom 2

            
        Returns
        -------
        pot : Numpy Array.
             Potential Energy
        """
        #Combine eps and sigma
        eps_ij = np.sqrt(eps_1 * eps_2)
        sigma_ij = (sigma_1 + sigma_2)/2

        #Calculate Potential
        pot = 4*eps_ij*( sigma_ij**12/(self.dist**12) - (sigma_ij**6)/(self.dist**6))

        return pot
    
    def genborn(self, q_1, born_1, q_2, born_2, eps=1, k_md = 138.935458):
        """
        Calculate Generalized-Born potential.

        Parameters
        ----------
        q_1 : Float.
            Charges on atom 1 
        born_1 : Float.
            Born radii on atom 1 
        q_2 : Float.
            Charges on atom 2
        born_2 : Float.
            Born radii on atom 2
        eps : float, optional
            Dielectric Constant. The default is 1.
        k_md : float, optional
            Coulomb Constant. The default is 138.935458 kJ/mol/nm. See ~ http://manual.gromacs.org/documentation/2019/reference-manual/definitions.html


        Returns
        -------
        pot : Numpy Array.
             Potential Energy
        """
        #Generalized Born Function
        f_gb = np.sqrt(self.dist**2 + born_1 * born_2
                       * np.exp(-self.dist**2/(4*born_1*born_2)) )
        
        #Calculate Potential
        pot_inter = (1/eps - 1) * k_md * (q_1 * q_2)/(f_gb)
        pot_self = 0.5 * (q_1**2/born_1 + q_2**2/born_2)
        pot = pot_inter + pot_self
        
        return pot
    
    def debye(self, q_1, r_1, c_1, q_2, r_2, c_2, eps=1, k_md = 138.935458, kbT = 2.494):
        """
        Calculate the Debye-Huckel Screened Coulomb Potential.

        Parameters
        ----------
        q_1 : Float.
            Charges on ion 1 
        r_1 : Float.
            Radius of ion 1
        c_1 : Float.
            Concentration of ion 1 in atoms per nm
        q_2 : Float.
            Charges on ion 2
        r_2 : Float.
            Radius of ion 2
        c_2 : Float.
            Concentration of ion 2 in atoms per nm
        eps : float, optional
            Dielectric Constant. The default is 1.
        k_md : float, optional
            Coulomb Constant. The default is 138.935458 kJ/mol/nm/e^2. See ~ http://manual.gromacs.org/documentation/2019/reference-manual/definitions.html
        kbT : float, optional
            Thermal energy at room temp. The default is 2.494 kJ/mol

        Returns
        -------
        pot : Numpy Array.
             Potential Energy
        """
        
        #Debye-Huckel Screening Length
        K = np.sqrt( k_md * 4 * np.pi/(eps * kbT) * ((q_1**2 * c_1) + (q_2**2 * c_2)) )
    
        # Screened Coulomb Part
        pot_1 = k_md * q_1 * q_2 / (eps * self.dist)
        # Debye damping part
        pot_2 = np.exp(-K * self.dist)
        # Additional damping term
        pot_3 = np.exp(K * (r_1 + r_2))/(1 + r_1 + r_2)
        # Total potential
        pot = pot_1 * pot_2 * pot_3
        
        return pot
    
    def sasa(self, r_1, r_2, gamma=2.092, r_w=0.17):
        """
        Calculate Solvent-Exclusion Potential.

        Parameters
        ----------
        r_1   : Float.
            Radius of ion 1
        r_2   : Float.
            Radius of ion 2
        gamma : Float.
            Surface Tension parameter. Default (water) is 2.092 kJ/mol/nm^2
        r_w   : Float.
            Radius of solvent. Default (water) is 0.17nm

        Returns
        -------
        pot : Numpy Array.
             Potential Energy
        """
        
        if type(self.dist) == float:
            if self.dist >= r_1 + r_2 + 2 * r_w:
                return 0
            term1 =  2*np.pi*(r_1 + r_w)**2
            term2 =  np.pi * self.dist * (r_1 + r_w)
            term3 =  np.pi * (r_1 + r_w)**3 / self.dist
            term4 = -np.pi * (r_2 + r_w)**2 * (r_1 + r_w) / self.dist
            
            pot = gamma * (term1 + term2 + term3 + term4)
            
        else:
            pot   =  np.zeros(self.dist.shape)
            term1 =  2*np.pi*(r_1 + r_w)**2
            term2 =  np.pi * self.dist * (r_1 + r_w)
            term3 =  np.pi * (r_1 + r_w)**3 / self.dist
            term4 = -np.pi * (r_2 + r_w)**2 * (r_1 + r_w) / self.dist
            
            pot = np.where(self.dist >= r_1 + r_2 + 2 * r_w, pot, 
                           gamma * (term1 + term2 + term3 + term4))
        
        return pot
    
    def tabulated(self, pot_x, x):
        """
        Tabulated potential. 

        Parameters
        ----------
        pot_x : Numpy Array.
            Potential at gridpoints.
        x : Numpy array.
            Gridpoints.

        Returns
        -------
        pot : Numpy Array. 
             Potential Energy
        """
        f = interp1d(x, pot_x, kind='linear', fill_value='extrapolate')
        pot = f(self.dist)
        return pot


class pairwise_forces(object):
    """
    Calculates various pairwise additive potential energies. By default outputs
    forces in kJ/mol/nm
    """    

    def __init__(self,diff,dist=None):
        """
        Initialize object by providing a set of distances between the atoms.

        Parameters
        ----------
        diff : Numpy Array. (npoints,ndim)
            Displacement vector between atoms
        dist : Numpy Array. (npoints,1)
            Distances between atoms. 
        """
        self.diff = diff
        if dist is None:
            self.dist = np.linalg.norm(diff,axis=-1)[:,np.newaxis]
        else:
            self.dist = dist
    
    def coulomb(self,q_1,q_2,eps=1,k_md = 138.935458):
        """
        Calculate Coulomb force.

        Parameters
        ----------
        q_1 : Float.
            Charges on ion 1 
        q_2 : Float.
            Charges on ion 2
        eps : float, optional
            Dielectric Constant. The default is 1.
        k_md : float, optional
            Coulomb Constant. The default is 138.935458 kJ/mol/nm. See ~ http://manual.gromacs.org/documentation/2019/reference-manual/definitions.html

        Returns
        -------
        frc : Numpy Array.
             Forces
        """
        frc = 1/eps * k_md * (q_1 * q_2)/(self.dist**3) * self.diff
        return frc
    
    def lj(self,eps_1,sigma_1,eps_2,sigma_2):
        """
        Calculate Lennard-Jones force.
        
        Parameters
        ----------
        eps_1 : Float.
            Epsilon param on ion 1 
        sigma_1 : Float.
            Sigma param on ion 1 
        eps_2 : Float.
            Epsilon param on ion 1 
        sigma_2 : Float.
            Sigma param on ion 2
            
        Returns
        -------
        frc : Numpy Array.
             Forces
        """
        #Combine eps and sigma
        eps_ij = np.sqrt(eps_1 * eps_2)
        sigma_ij = (sigma_1 + sigma_2)/2

        #Calculate Force
        frc = 48*eps_ij*( sigma_ij**12/(self.dist**14) - 0.5*(sigma_ij**6)/(self.dist**8)) * self.diff

        return frc
    
    def genborn(self, q_1, born_1, q_2, born_2, eps=1, k_md = 138.935458):
        """
        Calculate Generalized-Born potential.

        Parameters
        ----------
        q_1 : Float.
            Charges on ion 1 
        born_1 : Float.
            Born radii on ion 1 
        q_2 : Float.
            Charges on ion 2
        born_2 : Float.
            Born radii on ion 2
        eps : float, optional
            Dielectric Constant. The default is 1.
        k_md : float, optional
            Coulomb Constant. The default is 138.935458 kJ/mol/nm. See ~ http://manual.gromacs.org/documentation/2019/reference-manual/definitions.html


        Returns
        -------
        frc : Numpy Array.
             Forces
        """
        #Generalized Born Function
        f_gb = np.sqrt(self.dist**2 + born_2 * born_2
                       * np.exp(-self.dist**2/(4*born_1*born_2)) )
        
        #Calculate Force
        frc = (1/eps - 1) * k_md * (q_1 * q_2)/(f_gb**3) * \
        (1 - 0.25 * np.exp(-self.dist**2/(4*born_1*born_2)) ) * self.diff
        
        return frc
    
    def debye(self, q_1, r_1, c_1, q_2, r_2, c_2, eps=1, k_md = 138.935458, kbT = 2.494):
        """
        Calculate the Debye-Huckel Screened Coulomb Force.

        Parameters
        ----------
        q_1 : Float.
            Charges on ion 1 
        r_1 : Float.
            Raius of ion 1 
        c_1 : Float.
            Concentration of ion 1 in atoms per nm
        q_2 : Float.
            Charges on ion 2
        r_2 : Float.
            Raius of ion 2
        c_2 : Float.
            Concentration of ion 2 in atoms per nm
        eps : float, optional
            Dielectric Constant. The default is 1.
        k_md : float, optional
            Coulomb Constant. The default is 138.935458 kJ/mol/nm/e^2. See ~ http://manual.gromacs.org/documentation/2019/reference-manual/definitions.html
        kbT : float, optional
            Thermal energy at room temp. The default is 2.494 kJ/mol

        Returns
        -------
        frc : Numpy Array.
             Forces
        """
        
        #Debye-Huckel Screening Length
        K = np.sqrt( k_md * 4 * np.pi/(eps * kbT) * ((q_1**2 * c_1) + (q_2**2 * c_2)) )
    
        # Debye damping part
        damp = np.exp(-K * self.dist)
        
        # Coefficient
        coeff1 = k_md * q_1 * q_2 / (eps)
        coeff2 = np.exp(K * (r_1 + r_2))/(1 + r_1 + r_2)

        # Potential energy terms
        term1 = 1/(self.dist**3) * damp * self.diff 
        term2 = 1/(self.dist) * damp * self.diff
        
        # Total potential
        frc = coeff1 * coeff2 * (term1 + term2)
        
        return frc
    
    def sasa(self, r_1, r_2, gamma=2.092, r_w=0.17):
        """
        Calculate Solvent-Exclusion Force.

        Parameters
        ----------
        r_1   : Float.
            Radius of ion 1
        r_2   : Float.
            Radius of ion 2
        gamma : Float.
            Surface Tension parameter. Default (water) is 2.092 kJ/mol/nm^2
        r_w   : Float.
            Radius of solvent. Default (water) is 0.17nm

        Returns
        -------
        frc : Numpy Array.
             Forces
        """
        
        if type(self.dist) == float:
            if self.dist >= r_1 + r_2 + 2 * r_w:
                return 0
            term1 = -np.pi * (r_1 + r_w) / self.dist
            term2 =  np.pi * (r_1 + r_w)**3 / (self.dist**3)
            term3 = -np.pi * (r_2 + r_w)**2 * (r_1 + r_w) / (self.dist**3)
            
            frc = gamma * (term1 + term2 + term3) * self.diff
            
        else:
            pot   =  np.zeros(self.dist.shape)
            term1 = -np.pi * (r_1 + r_w) / self.dist
            term2 =  np.pi * (r_1 + r_w)**3 / (self.dist**3)
            term3 = -np.pi * (r_2 + r_w)**2 * (r_1 + r_w) / (self.dist**3)
            
            frc = np.where(self.dist >= r_1 + r_2 + 2 * r_w, pot, 
                            gamma * (term1 + term2 + term3) * self.diff)
        
        return frc
    
    def tabulated(self, frc_x, x):
        """
        Tabulated force. 

        Parameters
        ----------
        frc_x : Numpy Array.
            Force at gridpoints.
        x : Numpy array.
            Gridpoints.

        Returns
        -------
        frc : Numpy Array.
             Forces
        """
        f = interp1d(x, frc_x, kind='linear', fill_value='extrapolate')
        frc = f(self.dist) * self.diff/self.dist
        return frc
        

class ion_frc_iterator(pairwise_forces):
    """
    Iterates calculation of pairwise forces between a trajectory of monoatomic
    ions taken from simulation.
    """

    def __init__(self, pos, box_l):
        """
        Initialize class. Calculates distance/displacement between all 
        pairs of ions.

        Parameters
        ----------
        pos : numpy array.
            Trajectory of ion positions - (nt,nions,3)
        box_l : int
            Length of simulation box (assumes PBC dimensions are cubic)

        Methods
        ----------
        nt : int.
            Number of timesteps
        natom : int.
            Number of atoms/ions
        pdiff : list of numpy arrays - npairs x nt, 3
            Pairwise displacement vectors
        pdist : list of numpy arrays - npairs x nt
            Pairwise distances
        """
        # Instantiate Pairwise Force class
        pairwise_forces.__init__(self,0,0)
        
        # Pairwise Distant/Displacement calculation
        self.nt = np.size(pos,axis=0) #number of timesteps
        self.natom = np.size(pos, axis=1) #number of atoms
        self.box_l = box_l
        
        self.pdiff = []
        self.pdist = []
        for i in range(self.natom):
            for j in range(i+1,self.natom):
                #Calculate difference vector between charges
                diff = pos[:,i,:] - pos[:,j,:]

                #Adjust for PBC
                diff = np.where(diff > box_l/2, diff-box_l, diff)
                diff = np.where(diff < -box_l/2, diff+box_l, diff)

                #Calculate distance
                dist = np.linalg.norm(diff,axis=-1)[:,np.newaxis]

                #append to lists
                self.pdiff.append(diff)
                self.pdist.append(dist)      
        
    def calc_coulomb(self, q, cutoff, **kwargs):

        #list of all unique ij pairs
        ij_arr = list(combinations(range(self.natom),2))

        frc = np.zeros((self.nt,self.natom,3))

        #go through all unique pairs
        for k in range(len(ij_arr)):
            i,j = ij_arr[k]

            # calculate forces for pair ij
            pairwise_forces.__init__(self,self.pdiff[k], self.pdist[k])
            f_tmp = self.coulomb(q[i],q[j],**kwargs)

            # remove entries beyond cutoff
            if cutoff > 0:
                f_tmp = np.where(self.dist > cutoff, 0.0, f_tmp)

            # sum individual pair forces
            frc[:,i,:] += f_tmp
            frc[:,j,:] -= f_tmp

        return frc


    def calc_lj(self,eps,sigma,cutoff,**kwargs):

        #list of all unique ij pairs
        ij_arr = list(combinations(range(self.natom),2))

        frc = np.zeros((self.nt,self.natom,3))

        #go through all unique pairs
        for k in range(len(ij_arr)):
            i,j = ij_arr[k]

            # calculate forces for pair ij
            pairwise_forces.__init__(self,self.pdiff[k], self.pdist[k])
            f_tmp = self.lj(eps[i],sigma[i],eps[j],sigma[j],**kwargs)

            # remove entries beyond cutoff
            if cutoff > 0:
                f_tmp = np.where(self.dist > cutoff, 0.0, f_tmp)

            # sum individual pair forces
            frc[:,i,:] += f_tmp
            frc[:,j,:] -= f_tmp

        return frc

    def calc_genborn(self,q,born,cutoff,**kwargs):

        #list of all unique ij pairs
        ij_arr = list(combinations(range(self.natom),2))

        frc = np.zeros((self.nt,self.natom,3))

        #go through all unique pairs
        for k in range(len(ij_arr)):
            i,j = ij_arr[k]

            # calculate forces for pair ij
            pairwise_forces.__init__(self,self.pdiff[k], self.pdist[k])
            f_tmp = self.genborn(q[i],born[i],q[j],born[j],**kwargs)

            # remove entries beyond cutoff
            if cutoff > 0:
                f_tmp = np.where(self.dist > cutoff, 0.0, f_tmp)

            # sum individual pair forces
            frc[:,i,:] += f_tmp
            frc[:,j,:] -= f_tmp

        return frc
        
    def calc_debye(self,q,rad,cutoff,**kwargs):
        
        #concentrations of each ion assuming a electroneutral system
        conc = self.natom/(self.box_l**3)/2
        
        #list of all unique ij pairs
        ij_arr = list(combinations(range(self.natom),2))

        frc = np.zeros((self.nt,self.natom,3))

        #go through all unique pairs
        for k in range(len(ij_arr)):
            i,j = ij_arr[k]

            # calculate forces for pair ij
            pairwise_forces.__init__(self,self.pdiff[k], self.pdist[k])
            f_tmp = self.debye(q[i],rad[i],conc,q[j],rad[j],conc,**kwargs)

            # remove entries beyond cutoff
            if cutoff > 0:
                f_tmp = np.where(self.dist > cutoff, 0.0, f_tmp)

            # sum individual pair forces
            frc[:,i,:] += f_tmp
            frc[:,j,:] -= f_tmp
        return frc
    
    def calc_sasa(self,rad,cutoff,**kwargs):
        
        #list of all unique ij pairs
        ij_arr = list(combinations(range(self.natom),2))

        frc = np.zeros((self.nt,self.natom,3))

        #go through all unique pairs
        for k in range(len(ij_arr)):
            i,j = ij_arr[k]

            # calculate forces for pair ij
            pairwise_forces.__init__(self,self.pdiff[k], self.pdist[k])
            f_tmp = self.sasa(rad[i],rad[j],**kwargs)

            # remove entries beyond cutoff
            if cutoff > 0:
                f_tmp = np.where(self.dist > cutoff, 0.0, f_tmp)

            # sum individual pair forces
            frc[:,i,:] += f_tmp
            frc[:,j,:] -= f_tmp
        return frc
    
    def calc_tabulated(self, types, pot_cc, pot_aa, pot_ca, x, cutoff, **kwargs):
        #numerical differentiate potential functions
        f_cc = np.gradient(pot_cc,x)
        f_aa = np.gradient(pot_aa,x)
        f_ca = np.gradient(pot_ca,x)
        
        #list of all unique ij pairs
        ij_arr = list(combinations(range(self.natom),2))

        frc = np.zeros((self.nt,self.natom,3))

        #go through all unique pairs
        for k in range(len(ij_arr)):
            i,j = ij_arr[k]

            # calculate forces for pair ij
            pairwise_forces.__init__(self,self.pdiff[k], self.pdist[k])

            if (types[i] == "c" and types[j] == "a") or (types[i] == "a" and types[j] == "c"): 
                f_tmp = self.tabulated(f_ca,x)
            elif types[i] == "c" and types[j] == "c":
                f_tmp = self.tabulated(f_cc,x)
            elif types[i] == "a" and types[j] == "a":
                f_tmp = self.tabulated(f_aa,x)
            else:
                ValueError("type must either be 'c'(cation) or 'a' (anion) ")


            # remove entries beyond cutoff
            if cutoff > 0:
                f_tmp = np.where(self.dist > cutoff, 0.0, f_tmp)

            # sum individual pair forces
            frc[:,i,:] += f_tmp
            frc[:,j,:] -= f_tmp
        
        return frc