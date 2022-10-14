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
    assert np.size(mat,axis=0) == np.size(mat,axis=1)

    nt = np.size(t)
    ndim = np.size(mat,axis=0)
    indx = np.arange(ndim)
    
    
    eige, eigv = np.linalg.eig(mat)
    diag = np.zeros((nt, ndim,ndim),dtype=np.complex128)
    diag[:,indx,indx] = np.exp( np.outer(t,eige) )
                   
    mexp = eigv @ diag @ np.linalg.inv(eigv)
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
    assert np.size(mat,axis=0) == np.size(mat,axis=1)
    nt = np.size(t)
    ndim = np.size(mat,axis=0)
    indx = np.arange(ndim)
    
    eige, eigv = np.linalg.eigh(mat)
    
    diag = np.zeros((nt, ndim,ndim),dtype=np.float64)
    diag[:,indx,indx] = np.exp( np.outer(t,eige) )
                   
    mexp = eigv @ diag @ eigv.T
    return mexp

def matrixfunc(func, mat, t):
    """
    Calculate Time-Dependent Matrix Function

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
    assert np.size(mat,axis=0) == np.size(mat,axis=1)
    nt = np.size(t)
    ndim = np.size(mat,axis=0)
    indx = np.arange(ndim)
    
    
    eige, eigv = np.linalg.eig(mat)
    diag = np.zeros((nt, ndim,ndim),dtype=np.complex128)
    diag[:,indx,indx] = func( np.outer(t,eige) )
                   
    mfunc = eigv @ diag @ np.linalg.inv(eigv)
    return mfunc

def matrixfuncH(func, mat, t):
    """
    Calculate Time-Dependent Matrix Function of Hermitian Matrix

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
    assert np.size(mat,axis=0) == np.size(mat,axis=1)
    nt = np.size(t)
    ndim = np.size(mat,axis=0)
    indx = np.arange(ndim)
    
    
    eige, eigv = np.linalg.eigh(mat)
    diag = np.zeros((nt, ndim,ndim),dtype=np.float64)
    diag[:,indx,indx] = func( np.outer(t,eige) )
                   
    mfunc = eigv @ diag @ eigv.T
    return mfunc

def matrix_cos(modes, freq, t, mode="H"):
    """
    Calculate Time-Dependent Sine Function of Hermitian Matrix

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
    Ndof = np.size(freq)
    Nt = np.size(t)
    
    indx = np.arange(Ndof)
    M = np.zeros((Nt,Ndof,Ndof),dtype=np.float64)
    M[:,indx,indx] = np.cos( np.outer(t, freq ) ) 
    if mode in ["H", "h", "hermitian"]:
        M = modes @ M @ modes.T
    else:
        M = modes @ M @ np.linalg.inv(modes)
    return M

def matrix_sin(modes, freq, t, mode="H"):
    """
    Calculate Time-Dependent Sine Function of Hermitian Matrix

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
    Ndof = np.size(freq)
    Nt = np.size(t)
    
    indx = np.arange(Ndof)
    M = np.zeros((Nt,Ndof,Ndof),dtype=np.float64)
    M[:,indx,indx] = np.sin( np.outer(t, freq ) ) 
    if mode in ["H", "h", "hermitian"]:
        M = modes @ M @ modes.T
    else:
        M = modes @ M @ np.linalg.inv(modes)
    return M

########## Simulation Tools

def pbc_wrap(pos,box_dim):
    """
    Wrap Positions around a cubic periodic box

    Parameters
    ----------
    pos : Numpy array.
        positions.
    box_dim : Float.
        Length of side of periodix box.

    Returns
    -------
    pos : Numpy array.
        Wrapped positions.
    """
    
    pos = np.where( pos > box_dim, pos - box_dim, pos)
    pos = np.where( pos < 0, pos + box_dim, pos)
    return pos

class System(object):
    """ Creates a system """
    
    def __init__(self,m,nsys,ndim,box_dim):
        """
        Description of Methods:
        
        Simulation box methods
        ----------
        ndim : Number of spatial dimensions.
        box_dim : Size of simulation box.
        nsys : Number of particles in system.
        
        Masses - m, default = 1
        
        Positions/Velocities - pos,vel default = 0        
        """
        
        self.ndim = ndim
        self.nsys = nsys
        self.box_dim = box_dim

        if type(m) == float or type(m) == int or type(m) == np.float_:
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
        """ Randomly assign positions within box dimensions"""
        self.pos = (np.random.rand(self.nsys,self.ndim) - 0.5) * self.box_dim
        
            
    def set_pos_normal(self, mean, std):
        """ Assign positions of system from a normal distribution"""
        
        self.pos = np.random.normal(loc=mean,scale=std,size=(self.nsys,self.ndim))
        
    def set_pos_grid(self):
        """ Assign positions based on evenly spaced grid """
        
        #If number of spatial dimensions is 1.
        if self.ndim == 1:
            spc = self.box_dim[0]/self.nsys
            coords = np.linspace(-self.box_dim/2, self.box_dim/2 - spc, num=self.nsys )
            self.pos = coords
            
        #elif number of spatial dimensions is 2.
        elif self.ndim == 2:
            Nperdim = int(np.ceil(self.nsys**(0.5)))
            
            spc = self.box_dim[1]/Nperdim
            
            x = np.linspace(-self.box_dim[0]/2, self.box_dim[0]/2 - spc, num=Nperdim )
            y = np.linspace(-self.box_dim[0]/2, self.box_dim[0]/2 - spc, num=Nperdim )
            
            X,Y = np.meshgrid(x,y)
            coords = np.vstack([ X.ravel(), Y.ravel() ]).T
            coords = coords[0:self.nsys]
            
            self.pos = coords
        
        #elif number of spatial dimensions is 3.
        elif self.ndim == 3:
            Nperdim = int(np.ceil(self.nsys**(1.0/3.0)))
            
            spc = self.box_dim/Nperdim
            
            x = np.linspace(-self.box_dim[0]/2, self.box_dim[0]/2 - spc, num=Nperdim )
            y = np.linspace(-self.box_dim[1]/2, self.box_dim[1]/2 - spc, num=Nperdim )
            z = np.linspace(-self.box_dim[2]/2, self.box_dim[2]/2 - spc, num=Nperdim )
            
            X,Y,Z = np.meshgrid(x,y,z)
            coords = np.vstack([ X.ravel(), Y.ravel(), Z.ravel()]).T
            coords = coords[0:self.nsys]
            
            self.pos = coords
            
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
class NVE(object):
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
            x_t = pbc_wrap(x_t, self.system.box_dim)
        
        # Calculate Force at new position
        self.f_t = self.ff.calc_frc(x_t)

        # Move velocity half-step
        v_t = v_t + (self.dt/2.0 * self.f_t)/m

        # Update System Object
        self.system.pos = x_t.copy()
        self.system.vel = v_t.copy()
            
        
        return x_t, v_t, self.f_t
        
    
class Langevin(object):
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
            x_t = pbc_wrap(x_t, self.system.box_dim)
        
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

class GLD(object):
    """ 
    Class for integrating Generalized Langevin Dynamics.
    """
    
    def __init__(self,system, forcefield, dt, temp, As, Avs, Asv, Bs,  
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
        
        # Assign a bunch of useful variables
        self.nsys  = self.system.nsys
        self.ndim  = self.system.ndim
        self.naux  = np.size(As,axis=0)
        self.m = self.system.m
        
        # Set Force Array
        self.f_t = np.zeros((system.nsys, system.ndim),dtype=np.float64)
        
        #Set Bath Parameters
        self.set_bathparms( temp, As, Avs, Asv, Bs = Bs)
        
        #Set Bath State
        self.s_t = np.zeros((system.nsys, self.naux, system.ndim)
                          , dtype=np.float64)

        pass
            
    def set_bathparms(self, temp, As, Avs, Asv, Bs):
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
        self.Bs = Bs
        self.kbT = kb * temp
        
        # Ensure All dimensions Match
        self.naux = np.size(As,axis=0)
        
        assert self.naux == np.size(self.Bs,axis=0) == np.size(self.Bs,axis=1)
        assert self.naux == np.size(self.Avs,axis=1) == np.size(self.Asv,axis=0)
        assert 1 == np.size(self.Avs,axis=0) == np.size(self.Asv,axis=1)
                        
        pass
    
    def set_bathstate(self,s):
        """
        Set initial condition of GLE auxiliary variables
        """
        self.s_t = s
        
    def set_systemstate(self,x,v):
        """
        Set initial condition of system components
        """
        self.system.pos = x
        self.system.vel = v
        
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
            for k in range(len(self.reporters)):
                if i % self.reportints[k]  == 0:
                    self.reporters[k].save(x_t,v_t,a_t,f_t)
        pass     

    def step(self):
        """
        Single Step of Generalized Langevin dynamics integrator
        """
        
        # Create Position Array
        x_t = self.system.pos
        
        # Create Velocity Array
        v_t = self.system.vel
        
        #Move velocity half-step
        a_t = self.f_t/self.m - np.einsum("fj,njd->nd", self.Avs, self.s_t)
        v_t = v_t + self.dt/2.0 * a_t
        
        # Move Position full-step
        x_t = x_t + self.dt * v_t
        if self.PBC:
            x_t = pbc_wrap(x_t, self.system.box_dim)
            
        # Calculate Force at new position
        self.f_t = self.ff.calc_frc(x_t)
        
        # Move auxiliary variables full-step
        s_self = -np.einsum("ij,njd->nid", self.As, self.s_t)
        
        s_sys  = -np.einsum("if,nd->nid", self.Asv, v_t)
            
        noise = np.random.normal(loc=0.0, scale=1.0, size=(self.nsys,self.naux,self.ndim))
        s_ran = np.einsum("ij,njd->nid",self.Bs,noise) / np.sqrt(self.m[:,None])
            
        self.s_t = self.s_t + (self.dt * s_self) + (self.dt * s_sys) + (np.sqrt(self.dt) * s_ran)
    
        # Move velocity half-step
        a_t = self.f_t/self.m - np.einsum("fj,njd->nd", self.Avs, self.s_t)
        v_t = v_t + self.dt/2.0 * a_t
            
        # Update System Object
        self.system.pos = x_t.copy()
        self.system.vel = v_t.copy()
        
        # Function returns final state of auxiliary variables
        return x_t, v_t, a_t, self.f_t

class GLD_Aniso(object):
    """ 
    Class for integrating Generalized Langevin Dynamics.
    """
    
    def __init__(self,system, forcefield, dt, temp, bath_param_x, bath_param_y, 
                 bath_param_z, PBC = False, reporters = [], reportints = []):
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
        
        bath_param_x : List.
            Bath parameters for X-direction. Must contain 3-4 matrices: 
                As, Asv, Avs, and Bs
        bath_param_y : List.
            Bath parameters for Y-direction. Must contain 3-4 matrices: 
                As, Asv, Avs, and Bs
        bath_param_z : List.
            Bath parameters for Z-direction. Must contain 3-4 matrices: 
                As, Asv, Avs, and Bs
                
                
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
        
        # Assign a bunch of useful variables
        self.nsys  = self.system.nsys
        self.ndim  = 3
        
        self.m_2D = self.system.m.reshape(self.system.nsys,1)
        self.m_1D = self.system.m.flatten()
        
        # Set Force Array
        self.f_t = np.zeros((system.nsys, system.ndim),dtype=np.float64)
        
        # Set Bath Parameters
        self.set_bathparms( temp, bath_param_x, bath_param_y, bath_param_z)
        self.naux  = np.size(self.Bs_x,axis=0)

        # Set Bath State
        self.s_t = np.zeros((system.nsys, self.naux, 3), dtype=np.float64)
        pass
            
    def set_bathparms(self, temp, bath_param_x, bath_param_y, bath_param_z):
        """
        Set Ornstein-Uhlenbeck Bath Parameters. Assumes different GLE parameters
        for each axis
        
        K_x/y/z(t) = k0. Avs . exp(-As t) . Asv ~ scalar
        k0 ~ Bs Bs.T / (2 As + As.T) (when As and Bs commute)
        
        Parameters
        ----------
        bath_param_x : List.
            Bath parameters for X-direction. Must contain 3-4 matrices: 
                As, Asv, Avs, and Bs
        bath_param_y : List.
            Bath parameters for Y-direction. Must contain 3-4 matrices: 
                As, Asv, Avs, and Bs
        bath_param_z : List.
            Bath parameters for Z-direction. Must contain 3-4 matrices: 
                As, Asv, Avs, and Bs
        """
        
        self.kbT = kb * temp
        
        # Assign Parameters
        self.As_x, self.Asv_x, self.Avs_x, self.Bs_x = bath_param_x
        self.As_y, self.Asv_y, self.Avs_y, self.Bs_y = bath_param_y
        self.As_z, self.Asv_z, self.Avs_z, self.Bs_z = bath_param_z
        pass
    
    def set_bathstate(self,s):
        """
        Set initial condition of GLE auxiliary variables
        """
        self.s_t = s
        
    def set_systemstate(self,x,v):
        """
        Set initial condition of system components
        """
        self.system.pos = x
        self.system.vel = v
        
    def run(self, nt):
        """
        Run dynamics for nt steps.
        """

        #Initialize Forces
        self.f_t = self.ff.calc_frc(self.system.pos)
        
        # Report Initial State
        for k in range(len(self.reportints)):
            self.reporters[k].save(self.system.pos,self.system.vel,
                                   self.f_t/self.system.m,self.f_t)
            
        # Run Simulation and Report Data
        for i in range(nt):
            r_t, v_t, a_t, f_t = self.step()
        
            for k in range(len(self.reportints)):
                if i % self.reportints[k]  == 0:
                    self.reporters[k].save(r_t,v_t,a_t,f_t)
        pass     

    def step(self):
        """
        Single Step of Generalized Langevin dynamics integrator
        """
        # Create Position Array
        r_t = self.system.pos
        
        # Create Velocity Array
        v_t = self.system.vel
        
        # Create acceleration Array
        a_t = np.zeros((self.nsys,self.ndim),dtype=np.float64)
        
        #Move velocity half-step
        a_t[:,0] = self.f_t[:,0]/self.m_1D - np.einsum("fj,nj->n", self.Avs_x, self.s_t[:,:,0])
        a_t[:,1] = self.f_t[:,1]/self.m_1D - np.einsum("fj,nj->n", self.Avs_y, self.s_t[:,:,1])
        a_t[:,2] = self.f_t[:,2]/self.m_1D - np.einsum("fj,nj->n", self.Avs_z, self.s_t[:,:,2])
        v_t = v_t + self.dt/2.0 * a_t

        
        # Move Position full-step
        r_t = r_t + self.dt * v_t
        if self.PBC:
            r_t = pbc_wrap(r_t, self.system.box_dim)
            
        # Calculate Force at new position
        self.f_t = self.ff.calc_frc(r_t)
        
        ##############################################################
        ################# Move Auxiliary Variables #################
        ##############################################################
        
        noise = np.random.normal(loc=0.0, scale=1.0, size=(self.nsys,self.naux,3))

        # Move X auxiliary variables full-step
        s_self = -np.einsum("ij,nj->ni", self.As_x, self.s_t[:,:,0])
        s_sys  = -np.einsum("if,n->ni", self.Asv_x, v_t[:,0])
        s_ran = np.einsum("ij,nj->ni",self.Bs_x,noise[:,:,0]) / np.sqrt(self.m_2D)
            
        self.s_t[:,:,0] = self.s_t[:,:,0] + (self.dt * s_self) + (self.dt * s_sys) + (np.sqrt(self.dt) * s_ran)
        
        # Move Y auxiliary variables full-step
        s_self = -np.einsum("ij,nj->ni", self.As_y, self.s_t[:,:,1])
        s_sys  = -np.einsum("if,n->ni", self.Asv_y, v_t[:,1])
        s_ran = np.einsum("ij,nj->ni",self.Bs_y,noise[:,:,1]) / np.sqrt(self.m_2D)
            
        self.s_t[:,:,1] = self.s_t[:,:,1] + (self.dt * s_self) + (self.dt * s_sys) + (np.sqrt(self.dt) * s_ran)
        
        # Move Z auxiliary variables full-step
        s_self = -np.einsum("ij,nj->ni", self.As_z, self.s_t[:,:,2])
        s_sys  = -np.einsum("if,n->ni", self.Asv_z, v_t[:,2])
        s_ran = np.einsum("ij,nj->ni",self.Bs_z,noise[:,:,2]) / np.sqrt(self.m_2D)
            
        self.s_t[:,:,2] = self.s_t[:,:,2] + (self.dt * s_self) + (self.dt * s_sys) + (np.sqrt(self.dt) * s_ran)
        
        ##############################################################
        ##############################################################
        ##############################################################
        
        # Move velocity half-step
        a_t[:,0] = self.f_t[:,0]/self.m_1D - np.einsum("fj,nj->n", self.Avs_x, self.s_t[:,:,0])
        a_t[:,1] = self.f_t[:,1]/self.m_1D - np.einsum("fj,nj->n", self.Avs_y, self.s_t[:,:,1])
        a_t[:,2] = self.f_t[:,2]/self.m_1D - np.einsum("fj,nj->n", self.Avs_z, self.s_t[:,:,2])
        v_t = v_t + self.dt/2.0 * a_t
            
        # Update System Object
        self.system.pos = r_t.copy()
        self.system.vel = v_t.copy()
        
        # Function returns final state of auxiliary variables
        return r_t, v_t, a_t, self.f_t

class GLD_Harmonic(object):
    
    def __init__(self, system, nt, dt, Dpp, Dpq, Dqp, Dqq, posp_min, posq_min, 
                 posq0, velq0, PBC = False):
        """
  
        """
        
        #Set General Integration Parameters
        self.system = system
        self.nt = nt
        self.dt = dt
        self.PBC = PBC
        
        # Assign a bunch of useful variables
        self.nsys  = self.system.nsys
        self.ndim  = self.system.ndim
        self.nbath = np.size(Dqq,axis=0)
        self.m = self.system.m
        
        # Construct Time Array
        self.t_arr = np.arange(0,nt*dt,dt)
        
        # Set Bath Parameters
        self.set_bathparms(Dpp,Dqp,Dpq,Dqq)
        
        # Minimum Positions
        self.rp_min = posp_min
        self.rq_min = posq_min
        
        # Set Bath Initial Condition
        self.rq_0 = posq0
        self.vq_0 = velq0
        pass
            
    def set_bathparms(self,Dpp,Dqp,Dpq,Dqq):
        """

        """
        # Diagonalize Bath Hessian
        freq2, modes = np.linalg.eigh(Dqq)
        
        # Massage frequencies and inverse frequencies into arrays
        freq2 = np.where(freq2 > 0, freq2, 0)
        ifreq2 = np.where(freq2 == 0, 0, 1/freq2)
        freq = np.sqrt(freq2)
        ifreq = np.sqrt(ifreq2)
        
        if np.allclose(Dqq, Dqq.T, rtol=1e-8, atol=1e-12):
            imodes = modes.T
            cosWt = matrix_cos(modes, freq, self.t_arr)
            sinWt = matrix_sin(modes, freq, self.t_arr)

        else:
            imodes = np.linalg.inv(modes)
            cosWt = matrix_cos(modes, freq, self.t_arr,mode="A")
            sinWt = matrix_sin(modes, freq, self.t_arr,mode="A")

        # Coupling Matrices
        # C = imodes @ Dqp
        # iC = Dpq @ modes

        # Inverse Bath Hessian in normal mode and site basis
        iDqq = modes @ np.diag(ifreq2) @  imodes
        iW = modes @ np.diag(ifreq) @ imodes
    
        # Calculate relevant arrays for bath dynamics
        self.k_pmf = (Dpp - Dpq @ iDqq @ Dqp)*self.m
        self.K_mem = (Dpq @ cosWt @ iDqq @ Dqp)*self.m 
        self.k_rq0 = (Dpq @ cosWt)*self.m
        self.k_vq0 = (Dpq @ sinWt @ iW)*self.m
        self.k_rp0 = self.K_mem
        pass

    def conv_integral(self,kernel, signal, dt):
        """
        Compute a convolutional integral in real space.

        Parameters
        ----------
        kernel : Numpy Array.
            Integration Kernel.
        signal : Numpy Array.
            Signal to convolute.
        dt : float
            time difference.

        """
        K_flip = np.flip(kernel,axis=0)
        result = np.trapz(np.einsum("tij,tnj->tni",K_flip,signal),dx=dt,axis=0)
        return result
    
    def run(self):
        """
        Run dynamics for nt steps.
        """

        # Initial Condition
        r_t = self.system.pos
        v_t = self.system.vel
        
        # Arrays
        pos_array = [r_t]
        vel_array = [v_t]
        frc_array = []
        f1 = []
        f2 = []
        f3 = []
        
        # Run Simulation and Report Data
        for t in range(self.nt-1):
            
            # Calculate Forces
            f_det = -np.einsum("ij,nj->ni",self.k_pmf,r_t-self.rp_min)
            
            f_memory = -self.conv_integral(
                self.K_mem[0:t+1], np.array(vel_array), self.dt)
            
            f_rand = -( np.einsum("ij,j->i", self.k_vq0[t], self.vq_0) + 
                        np.einsum("ij,j->i", self.k_rq0[t], (self.rq_0 - self.rq_min) ) + 
                        np.einsum("ij,nj->ni", self.k_rp0[t], (pos_array[0] - self.rp_min) ) )

            f_t = f_det + f_memory + f_rand
            
            # Move Velocity half-step
            v_t = v_t + f_t*self.dt/(2.0*self.m)
            
            # Move Position full-step
            r_t = r_t + self.dt * v_t
            if self.PBC:
                r_t = pbc_wrap(r_t, self.system.box_dim)
                
            # Calculate Force at new position
            f_det = -np.einsum("ij,nj->ni",self.k_pmf,r_t-self.rp_min)
            
            f_t = f_det + f_memory + f_rand
            #print(f_det,f_memory,f_rand)
            
            # Move velocity another half-step
            v_t = v_t + f_t*self.dt/(2.0*self.m)
            
            #Append to lists
            pos_array.append(r_t)
            vel_array.append(v_t)
            frc_array.append(f_t)
            f1.append(f_det)
            f2.append(f_memory)
            f3.append(f_rand)

        # Update System Object
        self.system.pos = r_t.copy()
        self.system.vel = v_t.copy()
        pos_array = np.array(pos_array)
        vel_array = np.array(vel_array)
        frc_array = np.array(frc_array)
        f1 = np.array(f1)
        f2 =  np.array(f2)
        f3 = np.array(f3)
        
        # Function returns final state of auxiliary variables
        return pos_array, vel_array, frc_array, f1, f2, f3

if __name__ == "__main__":
    pass