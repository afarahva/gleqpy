 # -*- coding: utf-8 -*-
"""
GLE-Py
============

submodule: md
file: dynamics.py
author: Ardavan Farahvash (MIT)

description: 
Simple Python Implementation of Molecular Dynamics with Generalized Langevin
noise.
"""

import numpy as np
import scipy.sparse as sp

########## Calculation Tools
global kb
kb = 1

def matrixtransform(left, center, right):
    """
    Triple Matrix Product, M(t) = L @ C(t) @ R

    Parameters
    ----------
    left : Numpy Array (Ni,Nj)
        Left Matrix.
    center : Numpy Array. (Nt, Nj, Nk)
        Center Matrix, Time-Dependent
    right : Numpy Array
        Right Matrix.

    Returns
    -------
    product : Numpy Array. (Nt, Ni, Nk)
        Final product.
    """
    Nt = np.size(center,axis=0)
    product = np.zeros((Nt,left.shape[0],right.shape[1]),dtype=np.float64)
    for t in range(Nt):
        product[t,:,:] = left @ center[t] @ right
    return product

def matrixexp(mat,t):
    """
    Calculate Time-Dependent Matrix Exponential of a General Matrix

    Parameters
    ----------
    mat : Numpy Array. (ndim,dim)
        Matrix.
    t : Numpy Array. (nt)
        Timesteps.

    Returns
    -------
    mexp : Numpy Array. (nt,ndim,ndim)
        Matrix Exponential.
    """
    nt = np.size(t)
    ndim = np.size(mat,axis=0)
    assert np.size(mat,axis=0) == np.size(mat,axis=1) 
    dindx = np.diag_indices(ndim)
    
    eige, eigv = np.linalg.eig(mat)
    
    diag = np.zeros((nt, ndim,ndim),dtype=np.complex128)
    diag[:,dindx[0],dindx[1]] = np.exp( np.outer(t,eige) )
                   
    mexp = matrixtransform(eigv, diag, np.linalg.inv(eigv))
    return mexp

def matrixexpH(mat,t):
    """
    Calculate Time-Dependent Matrix Exponential of a Hermitian Matrix

    Parameters
    ----------
    mat : Numpy Array. (ndim,dim)
        Matrix.
    t : Numpy Array. (nt)
        Timesteps.

    Returns
    -------
    mexp : Numpy Array. (nt,ndim,ndim)
        Matrix Exponential.
    """
    nt = np.size(t)
    ndim = np.size(mat,axis=0)
    assert np.size(mat,axis=0) == np.size(mat,axis=1)
    dindx = np.diag_indices(ndim)
    
    eige, eigv = np.linalg.eigh(mat)
    
    diag = np.zeros((nt, ndim,ndim),dtype=np.complex128)
    diag[:,dindx[0],dindx[1]] = np.exp( np.outer(t,eige) )
                   
    mexp = matrixtransform(eigv, diag, eigv.T)
    return mexp

########## Simulation Tools

def PBCwrap(pos,box_l):
    """
    Wrap Positions around a cubic periodic box

    Parameters
    ----------
    pos : Numpy array.
        positions.
    box_l : Float.
        Length of side of periodix box.

    Returns
    -------
    pos : Numpy array.
        Wrapped positions.
    """
    
    pos = np.where( pos > box_l, pos - box_l, pos)
    pos = np.where( pos < 0, pos + box_l, pos)
    return pos

class mk_system(object):
    """ Creates a system """
    
    def __init__(self,m,nsys,ndim,box_l):
        """
        Description of Methods:
        
        Simulation box methods
        ----------
        ndim : Number of spatial dimensions.
        box_l : Size of simulation box.
        nsys : Number of particles in system.
        
        Masses - m, default = 1
        
        Positions/Velocities - pos,vel default = 0        
        """
        
        self.ndim = ndim
        self.nsys = nsys
        self.box_l = box_l

        if type(m) == float or type(m) == int:
            self.m = np.ones((nsys,1),dtype=np.float64) * m
        elif type(m) == np.ndarray:
            self.m = m.reshape(nsys,1)
        else:
            raise TypeError("m must be scalar or np.ndarray")
        
        self.pos = np.zeros((nsys,ndim))
        self.vel = np.zeros((nsys,ndim))

        
    def set_vel_to_temp(self,beta):
        """ Set velocities to temperature beta = 1/(k_b T)"""
        vel_std = np.repeat(np.sqrt(1/(self.m*beta)) , self.ndim, axis=1)
        
        self.vel = np.random.normal(loc=0.0, scale=vel_std, 
                                    size=(self.nsys,self.ndim))
        
    def set_pos_random(self):
        """ Randomly assign positions """
        self.pos = (np.random.rand(self.nsys,self.ndim) - 0.5) * self.box_l
        
    def set_pos_sys_grid(self):
        """ Assign system positions based on evenly spaced grid """
        
        #If number of spatial dimensions is 1.
        if self.ndim == 1:
            spc = self.box_l/self.nsys
            coords = np.linspace(-self.box_l/2, self.box_l/2 - spc, num=self.nsys )
            self.pos = coords
            
        #elif number of spatial dimensions is 2.
        elif self.ndim == 2:
            Nperdim = int(np.ceil(self.nsys**(0.5)))
            
            spc = self.box_l/Nperdim
            
            x = np.linspace(-self.box_l/2, self.box_l/2 - spc, num=Nperdim )
            y = np.linspace(-self.box_l/2, self.box_l/2 - spc, num=Nperdim )
            
            X,Y = np.meshgrid(x,y)
            coords = np.vstack([ X.ravel(), Y.ravel() ]).T
            coords = coords[0:self.nsys]
            
            self.pos = coords
        
        #elif number of spatial dimensions is 3.
        elif self.ndim == 3:
            Nperdim = int(np.ceil(self.nsys**(1.0/3.0)))
            
            spc = self.box_l/Nperdim
            
            x = np.linspace(-self.box_l/2, self.box_l/2 - spc, num=Nperdim )
            y = np.linspace(-self.box_l/2, self.box_l/2 - spc, num=Nperdim )
            z = np.linspace(-self.box_l/2, self.box_l/2 - spc, num=Nperdim )
            
            X,Y,Z = np.meshgrid(x,y,z)
            coords = np.vstack([ X.ravel(), Y.ravel(), Z.ravel()]).T
            coords = coords[0:self.nsys]
            
            self.pos = coords
            
        else:
            raise ValueError("Only integers 1, 2, or 3 are allowed for ndim")
            
class reporter_Temp(object):
    """
    Saves Position, Velocity, Acceleration, and per particle deterministic
    force at each timestep. 
    """
    def __init__(self,kb,m,ndim):
        self.ndim = ndim
        self.kb = kb
        self.m = m
        self.temp_arr = []
        pass
        
    def save(self,x,v,a,f):
        v_sq = np.sum( np.mean( v**2, axis=0) )
        temp = self.m/(self.ndim * self.kb) * v_sq
        self.temp_arr.append(temp)
        pass
        
    def output(self):
        temp_out = np.array(self.temp_arr)
        return temp_out

class reporter_PosVel(object):
    """
    Saves Position and Velocity of each particle at each timestep.
    """
    def __init__(self):
        self.x_arr = []
        self.v_arr = []
        pass
        
    def save(self,x,v,a,f):
        self.x_arr.append(x)
        self.v_arr.append(v)
        pass
        
    def output(self):
        x_out = np.array(self.x_arr)
        v_out = np.array(self.v_arr)
        return x_out, v_out
    
class reporter_PosVelAccFrc(object):
    """
    Saves Position, Velocity, Acceleration, and per particle deterministic
    force at each timestep. 
    """
    def __init__(self):
        self.x_arr = []
        self.v_arr = []
        self.a_arr = []
        self.f_arr = []
        pass
        
    def save(self,x,v,a,f):
        self.x_arr.append(x)
        self.v_arr.append(v)
        self.a_arr.append(a)
        self.f_arr.append(f)
        pass
        
    def output(self):
        x_out = np.array(self.x_arr)
        v_out = np.array(self.v_arr)
        a_out = np.array(self.a_arr)
        f_out = np.array(self.f_arr)
        return x_out, v_out, a_out, f_out

def calc_ou_var(B, A, nt, dt, Avs = None, Asv = None, method="full"):
    """
    Calculate the stationary variance of the Multivariate Ornstein-Uhlenbeck Process

    Parameters
    ----------
    B : Numpy Array. - (naux,naux)
        White Noise Multiplier Matrix
    A : Numpy Array. - (naux,naux)
        Deterministic Evolution matrix
    nt : Int.
        Number of Simulation Timesteps.
    dt : Float.
        Step length.
        
    Avs : Numpy Array, optional - (nsys,naux)
        Auxiliary -> Momentum Projeciton Matrix. The defualt is None.
    Asv : Numpy Array, optional - (nsys,naux)
        Momentum -> Auxiliary Projeciton Matrix. The defualt is None.
    method : String, optional
        How to calculate MSD. The default is "full".

    Returns
    -------
    statvar : Numpy Array. - (nsys,nsys)
        Variance of projected Ornstein-Uhlenbeck process.
    """
    
    tmax = nt * dt
    t_arr = np.arange(0,tmax,dt)
    
    naux  = np.size(B, axis=0)
        
    if type(Avs) != np.ndarray:
        Avs = np.identity(naux)
        
    if type(Asv) != np.ndarray:
        Asv = np.identity(naux)
    
    #If operators A, A.T, B, and B.T don't commute, use Lyapunov Formula
    # c = Avs @ \int_0^t e^(-A t) B B.T e^(-A.T t) dt @ Asv
    if method == "full":
            # Calculate Matrix Exponential
            A_mexp = matrixexp(-A,t_arr)
            
            # Calculate B @ B.T
            B_sq = B @ B.T
            integrand = A_mexp @ B_sq @ A_mexp.swapaxes(1,2)
            var0 = np.trapz(integrand, t_arr, axis=0)
            statvar = np.real( Avs @ var0 @ Asv ) # stationary variance
        
    #If operators A, A.T, B, and B.T do commute
    elif method == "commuting":
            B_sq = B.dot(B.T)
            A_inv = np.linalg.solve(A + A.T, np.identity(naux))
            var0  = A_inv @ B_sq 
            statvar = Avs @ var0 @ Asv # stationary variance
        
    else:
        raise ValueError("method argument must be either 'full' or 'commuting' ")
        
    return statvar

########## Simulation Integrators
class nve(object):
    """
    Class for integrating microcanonical dynamics.
    """
    def __init__(self, system, forcefield, dt, PBC=False, reporters = [], reportints = []):
        """
        Parameters
        ----------
        system : System Object.
        forcefield : Forcefield Object.
        dt : Float
            Length of each timestep.
        PBC : Bool. 
            Whether or not to use periodic boundary conditions. (Default = False)
        
        reporters : list of reporter objects, optional
        reportints : list of reporter intervals, optional
        """
        #Set simulation parameters
        self.system = system
        self.ff = forcefield
        self.dt = dt
        self.PBC = PBC
        self.reporters = reporters
        self.reportints = reportints
        
        # Set Force Array
        self.f_t = np.zeros((system.nsys, system.ndim),dtype=np.float64)
        
        pass
    
    def run(self, nt):
        """
        Run calculation for nt steps.
        """
        
        print("Running Simulation")
        
        #Initial Forces
        self.f_t = self.ff.calc_frc(self.system.pos)
        
        # Report Initial State
        for k in range(len(self.reportints)):
            self.reporters[k].save(self.system.pos,self.system.vel,
                                   self.f_t/self.system.m,self.f_t)
        
        for i in range(nt):
            self.step()
        
            # Report Simulation Data
            for k in range(len(self.reportints)):
                if i % self.reportints[k]  == 0:
                    self.reporters[k].save(self.system.pos,self.system.vel,
                                           self.f_t/self.system.m,self.f_t)
        pass
    
    def step(self):
        """
        Single Step of Velocity Verlet method. 
    
        Scheme:
        v_t+1/2 = v_t + a_t * dt/2
        x_t+1 = x_t + v_t*dt
        v_t+1 = v_t+1/2 + a_t+1 +*dt/2
        
        """

        # mass
        m = self.system.m
        
        # Create Position Array
        x_t = self.system.pos
        
        # Create Velocity Array
        v_t = self.system.vel
        
        # Move velocity half-step            
        v_t = v_t + (self.dt/2.0 * self.f_t)/m
        
        # Move Position full-step
        x_t = x_t + self.dt * v_t
        
        if self.PBC:
            x_t = PBCwrap(x_t, self.system.box_l)
        
        # Calculate Force at new position
        self.f_t = self.ff.calc_frc(x_t)

        # Move velocity half-step
        v_t = v_t + (self.dt/2.0 * self.f_t)/m

        # Update System Object
        self.system.pos = x_t.copy()
        self.system.vel = v_t.copy()
            
        
        return x_t, v_t, self.f_t
        
    
class langevin(object):
    """
    Class for integrating langevin dynamics.
    """
    def  __init__(self, system, forcefield, dt, temp, friction, rancoeff=None, 
                  PBC=False, reporters=[], reportints=[]):
        """
        Parameters
        ----------
        system : System Object.
        forcefield : Forcefield Object.
        dt : Float
            Length of each timestep.
        temp : Float. 
            Temperature.
            
        friction : Float or 1D Numpy Array. 
            Friction Coefficient
            
        rancoeff : Float or 1D array, optional 
            Random Force coefficient, default set using fluctuation-dissipation thm.
            
        PBC : Bool. 
            Whether or not to use periodic boundary conditions. (Default = False)
            
        reporters : list of reporter objects, optional
        reportints : list of reporter intervals, optional
        """
        # Set simulation parameters
        self.system = system
        self.ff = forcefield
        self.dt = dt
        self.PBC = PBC
        self.reporters = reporters
        self.reportints = reportints
        
        # Set Force Array
        self.f_t = np.zeros((system.nsys, system.ndim),dtype=np.float64)
        
        # Set Langevin bath parameters
        
        # Thermal Energy
        self.kbT = temp * kb
        
        # Set Friction Coefficient Array: A
        if type(friction) != np.ndarray: # if argument is a float
            self.A = np.ones(system.nsys) * friction
        else:                            # if argument is an array
            assert system.nsys == np.size(self.A)
            
        # Set Random Force Coefficients: B
        # if argument == None, choose B such that it follows F-D thm.
        if np.all(rancoeff) == None:
            self.B = np.sqrt(2 * self.kbT * self.A)
            
        self.A = self.A[:,None]
        self.B = self.B[:,None]
        pass
            
    def run(self, nt):
        """
        Run Calculations for nt steps.
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
        Single Step of Langevin integrator. 
        """
        
        nsys  = self.system.nsys
        ndim  = self.system.ndim
        dt = self.dt
        
        # mass
        m = self.system.m
        
        # Create Position Array
        x_t = self.system.pos
        
        # Create Velocity Array
        v_t = self.system.vel
                
        # Move velocity half-step
        noise = np.random.normal(loc=0.0, scale=1, size=(nsys,ndim))
        stoch = self.B * noise
        det = self.f_t - self.A * v_t
            
        v_t = v_t + (dt/2.0 * det)/m + (np.sqrt(dt)/2 * stoch)/m
                    
        # Move Position full-step
        x_t = x_t + self.dt * v_t
        
        if self.PBC:
            x_t = PBCwrap(x_t, self.system.box_l)
        
        # Calculate Force at new position
        self.f_t = self.ff.calc_frc(x_t)

        # Move velocity half-step
        det = self.f_t - self.A * v_t
        a_t = det/m + stoch/m
        v_t = v_t + (dt/2.0 * det)/m + (np.sqrt(dt)/2 * stoch)/m
            
        # Update System Object
        self.system.pos = x_t.copy()
        self.system.vel = v_t.copy()
        
        return x_t, v_t, a_t, self.f_t

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


if __name__ == "__main__":
    pass