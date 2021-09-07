 # -*- coding: utf-8 -*-
"""
GLEPy
============

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
        self.forcefield = forcefield
        self.dt = dt
        self.PBC = PBC
        self.reporters = reporters
        self.reportints = reportints
        pass
    
    def run(self, nt):
        """
        Run calculation for nt steps.
        """
        self.verlet(self.system, self.forcefield, nt, self.dt, 
                           PBC = self.PBC, reporters = self.reporters, 
                           reportints = self.reportints)
        pass
    
    @staticmethod
    def verlet(system, forcefield, nt, dt, PBC=False, reporters = [], reportints = []):
        """
        Velocity Verlet algorithm. 
    
        Parameters
        ----------
        system : System Object.
        forcefield : Forcefield Object.
        nt : Int
            Number of timesteps.
        dt : Float
            Length of each timestep.
        
        PBC : Bool. (Optional)
            Whether or not to use periodic boundary conditions.
        reporters : list of reporter objects, optional
        reportints : list of reporter intervals, optional
        
        Returns
        -------
        reporters : list of reporter objects
            Data saved from simulation in specified reporter objects.
        """
        # system and dimension
        nsys  = system.nsys
        ndim  = system.ndim
        
        # mass
        m = system.m
        
        # Create Position Array
        x_t = system.pos
        
        # Create Velocity Array
        v_t = system.vel
        
        # Create Force Arrays
        ff = forcefield
        f_t = ff.calc_frc(x_t)
            
        # Create Total Force Arrays
        a_t = np.zeros((nsys,ndim))
            
        # Save Initial State
        for k in range(len(reportints)):
            reporters[k].save(x_t,v_t,a_t,f_t)
            
        # Run Simulation
        print("Running Simulation")
        for i in range(1,nt):
            
            # Move velocity half-step            
            v_t_half = v_t + (dt/2.0 * f_t)/m
            
            # Move Position full-step
            x_t = x_t + dt * v_t_half
            if PBC:
                x_t = PBCwrap(x_t, system.box_l)
            
            # Calculate Force at new position
            f_t = ff.calc_frc(x_t)
    
            # Move velocity full-step
            v_t = v_t_half + (dt/2.0 * f_t)/m
            
            # Report
            for k in range(len(reportints)):
                if i % reportints[k]  == 0:
                    reporters[k].save(x_t,v_t,f_t/m,f_t)
        
        # Save Final Simulation State to System Object
        system.pos = x_t.copy()
        system.vel = v_t.copy()
        
        pass
        
    
class langevin(object):
    """
    Class for integrating langevin dynamics.
    """
    def  __init__(self, system, forcefield, dt, temp, friction, rancoeff=None, 
                  PBC=False, style=1, reporters=[], reportints=[]):
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

        PBC : Bool. 
            Whether or not to use periodic boundary conditions. (Default = False)
        reporters : list of reporter objects, optional
        reportints : list of reporter intervals, optional
        """
        # Set simulation parameters
        self.system = system
        self.forcefield = forcefield
        self.dt = dt
        self.PBC = PBC
        self.reporters = reporters
        self.reportints = reportints
        self.style = style
        
        # Set Langevin bath parameters
        
        # Thermal Energy
        self.kbT = temp * kb
        
        # Set Friction Coefficient Array: A
        if type(friction) != np.ndarray:
            self.A = np.ones(system.nsys) * friction
        else:
            assert system.nsys == np.size(self.A)
            
        # Set Random Force Coefficients: B 
        if np.all(rancoeff) == None:
            self.B = np.sqrt(2 * self.kbT * self.A)
        pass
            
    def run(self, nt):
        """
        Run Calculations for nt steps.
        """
        
        self.verlet(self.system, self.forcefield, nt, self.dt, self.kbT, 
                    self.A[:,None], self.B[:,None], PBC = self.PBC, 
                    reporters = self.reporters, reportints = self.reportints)
        pass
                
    @staticmethod     
    def verlet(system, forcefield, nt, dt, kbT, A, B, PBC = False, reporters = [], reportints = []):
        """
        Langevin Dynamics with Velocity Verlet algorithm. 
    
        Parameters
        ----------
        system : System Object.
        forcefield : Forcefield Object.
        nt : Int
            Number of timesteps.
        dt : Float
            Length of each timestep.
        kbT : Float.
            Thermal Energy.
        A : Numpy Array or Float.
            Markovian Friction Kernel.
        B : Numpy Array or Float.
            Wiener Multiplier Matrix
        PBC : Bool. 
            Whether or not to use periodic boundary conditions.
            
        reporters : list of reporter objects, optional
        reportints : list of reporter intervals, optional
        
        Returns
        -------
        reporters : list of reporter objects
            Data saved from simulation in specified reporter objects.
        """
        nsys  = system.nsys
        ndim  = system.ndim
        
        # mass
        m = system.m
        
        # Create Position Array
        x_t = system.pos
        
        # Create Velocity Array
        v_t = system.vel
        
        # Create Force Arrays
        ff = forcefield
        f_t = ff.calc_frc(x_t)
            
        # Create Total Force Arrays
        a_t = np.zeros((nsys,ndim))
            
        # Save Initial State
        for k in range(len(reportints)):
            reporters[k].save(x_t,v_t,a_t,f_t)
            
        # Run Simulation
        print("Running Simulation")
        for i in range(1,nt):
            
            # Move velocity half-step
            noise = np.random.normal(loc=0.0, scale=1, size=(nsys,ndim))
            stoch = B * noise
            det = f_t - A * v_t
            
            v_t_half = v_t + (dt/2.0 * det)/m + (np.sqrt(dt)/2 * stoch)/m
            
            # Move Position full-step
            x_t = x_t + dt * (v_t_half - v_t_half.mean(axis=0))
            if PBC:
                x_t = PBCwrap(x_t, system.box_l)
            
            # Calculate Force at new position
            f_t = ff.calc_frc(x_t)
    
            # Move velocity full-step
            det = f_t - A * v_t_half
            a_t = det/m + stoch/m
            v_t = v_t_half + (dt/2.0 * det)/m + (np.sqrt(dt)/2 * stoch)/m
                            
            # Report
            for k in range(len(reportints)):
                if i % reportints[k]  == 0:
                    reporters[k].save(x_t,v_t,a_t,f_t)
        
        # Save Simulation State to System Object
        system.pos = x_t.copy()
        system.vel = v_t.copy()
        
        pass
    
    @staticmethod
    def verlet_corr(system, forcefield, nt, dt, kbT, A, B, PBC = False, reporters = [], reportints = []):
        """
        Langevin Dynamics with Verlet hopping algorithm. Allows for bath to induce
        correlations between particles.
    
        Parameters
        ----------
        system : System Object.
        forcefield : Forcefield Object.
        nt : Int
            Number of timesteps.
        dt : Float
            Length of each timestep.
        kbT : Float.
            Thermal Energy.
        A : 1D Numpy Array or Float.
            Markovian Friction Kernel.
        B : 1D Numpy Array or Float.
            Wiener Multiplier Matrix
        PBC : Bool. 
            Whether or not to use periodic boundary conditions.
            
        reporters : list of reporter objects, optional
        reportints : list of reporter intervals, optional
        
        Returns
        -------
        reporters : list of reporter objects
            Data saved from simulation in specified reporter objects.
        """
        nsys  = system.nsys
        ndim  = system.ndim
        
        # mass
        m = system.m
        
        # Create Position Array
        x_t = system.pos
        
        # Create Velocity Array
        v_t = system.vel
        
        # Create Force Arrays
        ff = forcefield
        f_t = ff.calc_frc(x_t)
            
        # Create Total Force Arrays
        a_t = np.zeros((nsys,ndim))
            
        # Save Initial State
        for k in range(len(reportints)):
            reporters[k].save(x_t,v_t,a_t,f_t)
            
        # Run Simulation
        print("Running Simulation")
        for i in range(1,nt):
            
            # Move velocity half-step
            noise = np.random.normal(loc=0.0, scale=(1/np.sqrt(ndim)), size=(nsys,ndim))
            stoch = B.dot(noise)
            det = f_t - A.dot(v_t)
            
            v_t_half = v_t + (dt/2.0 * det)/m + (np.sqrt(dt/2.0) * stoch)/m
            
            # Move Position full-step
            x_t = x_t + dt * v_t_half
            if PBC:
                PBCwrap(x_t, system.box_l)
            
            # Calculate Force at new position
            f_t = ff.calc_frc(x_t)
    
            # Move velocity full-step
            det = f_t - A.dot(v_t_half)
            a_t = det/m + stoch/m
            v_t = v_t_half + (dt/2.0 * det)/m + (np.sqrt(dt/2.0) * stoch)/m
                            
            # Report
            for k in range(len(reportints)):
                if i % reportints[k]  == 0:
                    reporters[k].save(x_t,v_t,a_t,f_t)
        
        # Save Simulation State to System Object
        system.pos = x_t
        system.vel = v_t
        
        return reporters

class langevin_corr(object):
    """
    Class for integrating langevin dynamics with a correlatted bath. 
    """
    def  __init__(self, system, forcefield, dt, temp, friction, rancoeff=None, 
                  PBC=False, style=1, reporters=[], reportints=[]):
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

        PBC : Bool. 
            Whether or not to use periodic boundary conditions. (Default = False)
        reporters : list of reporter objects, optional
        reportints : list of reporter intervals, optional
        """
        # Set simulation parameters
        self.system = system
        self.forcefield = forcefield
        self.dt = dt
        self.PBC = PBC
        self.reporters = reporters
        self.reportints = reportints
        self.style = style
        
        # Set Langevin bath parameters
        
        # Thermal Energy
        self.kbT = temp * kb
        
        # Set Friction Coefficient Array: A
        assert self.system.nsys == np.size(self.A, axis=0) == np.size(self.A, axis=1)
        self.A = friction
        
        # Set Random Force Coefficients: B 
        if np.all(rancoeff) == None:
            self.B = np.sqrt(self.kbT) * np.linalg.cholesky(self.A + self.A.T)
        pass
        

    def run(self, nt):
        """
        Run Calculations for nt steps.
        """
        self.verlet_corr(self.system, self.forcefield, nt, self.dt, self.kbT, 
                         self.A, self.B, PBC = self.PBC, reporters = self.reporters, reportints = self.reportints)
           
        pass
    
    @staticmethod
    def verlet_corr(system, forcefield, nt, dt, kbT, A, B, PBC = False, reporters = [], reportints = []):
        """
        Langevin Dynamics with Verlet hopping algorithm. Allows for bath to induce
        correlations between particles.
    
        Parameters
        ----------
        system : System Object.
        forcefield : Forcefield Object.
        nt : Int
            Number of timesteps.
        dt : Float
            Length of each timestep.
        kbT : Float.
            Thermal Energy.
        A : 1D Numpy Array or Float.
            Markovian Friction Kernel.
        B : 1D Numpy Array or Float.
            Wiener Multiplier Matrix
        PBC : Bool. 
            Whether or not to use periodic boundary conditions.
            
        reporters : list of reporter objects, optional
        reportints : list of reporter intervals, optional
        
        Returns
        -------
        reporters : list of reporter objects
            Data saved from simulation in specified reporter objects.
        """
        nsys  = system.nsys
        ndim  = system.ndim
        
        # mass
        m = system.m
        
        # Create Position Array
        x_t = system.pos
        
        # Create Velocity Array
        v_t = system.vel
        
        # Create Force Arrays
        ff = forcefield
        f_t = ff.calc_frc(x_t)
            
        # Create Total Force Arrays
        a_t = np.zeros((nsys,ndim))
            
        # Save Initial State
        for k in range(len(reportints)):
            reporters[k].save(x_t,v_t,a_t,f_t)
            
        # Run Simulation
        print("Running Simulation")
        for i in range(1,nt):
            
            # Move velocity half-step
            noise = np.random.normal(loc=0.0, scale=(1/np.sqrt(ndim)), size=(nsys,ndim))
            stoch = B.dot(noise)
            det = f_t - A.dot(v_t)
            
            v_t_half = v_t + (dt/2.0 * det)/m + (np.sqrt(dt/2.0) * stoch)/m
            
            # Move Position full-step
            x_t = x_t + dt * v_t_half
            if PBC:
                PBCwrap(x_t, system.box_l)
            
            # Calculate Force at new position
            f_t = ff.calc_frc(x_t)
    
            # Move velocity full-step
            det = f_t - A.dot(v_t_half)
            a_t = det/m + stoch/m
            v_t = v_t_half + (dt/2.0 * det)/m + (np.sqrt(dt/2.0) * stoch)/m
                            
            # Report
            for k in range(len(reportints)):
                if i % reportints[k]  == 0:
                    reporters[k].save(x_t,v_t,a_t,f_t)
        
        # Save Simulation State to System Object
        system.pos = x_t.copy()
        system.vel = v_t.copy()
        pass
    
class gld(object):
    """ 
    Class for integrating the Generalized Langevin Dynamics.
    """
    
    def __init__(self,system, forcefield, dt, style=1, PBC = False, reporters = [], reportints = []):
        """
        Parameters
        ----------
        system : System Object.
        forcefield : Forcefield Object.
        nt : Int.
            Number of timesteps.
        dt : Float.
            Length of each timestep.d
        
        Optional
        ----------
        style : String. 
            Style option for integrator, can be one of: ou, ou-corr, 
            or ou-sprs-corr.
        PBC : Bool. 
            Whether or not to use periodic boundary conditions, Default=False
        s0 : Float or Numpy Array. 
            Initial positions of bath particles. Default is 0
        reporters : list of reporter objects
        reportints : list of reporter intervals

        """
        
        #Set General Integration Parameters
        self.system = system
        self.forcefield = forcefield
        self.dt = dt
        self.style = style
        self.PBC = PBC
        self.reporters = reporters
        self.reportints = reportints
        
        # Set Ornstein-Uhlenbeck Back Parameters
        self.As = None
        self.Bs = None
        self.Asv = None
        self.Avs = None
        self.kbT = 0
        self.k0 = None
        pass
        
    def set_bath_params(self,As, Bs, Asv, Avs, temp, k0=None):
        """
        Set Bath Parameters.

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
        self.Bs = Bs
        self.Asv = Asv
        self.Avs = Avs
        self.kbT = kb * temp
        
        # Ensure Dimensions Match
        naux  = np.size(Bs,axis=0)
        
        assert naux == np.size(Bs,axis=1) == np.size(As,axis=0) == np.size(As,axis=1)
        assert naux == np.size(Avs,axis=1) == np.size(Asv,axis=0)
        assert 1 == np.size(Avs,axis=0) == np.size(Asv,axis=1)
            
        # If k0 is not assigned, assume it is identity or already folded into Asv
        if np.all(k0) != None:
            self.Asv = k0 @ self.Asv
            
        pass
            
        
    def run(self, nt):
        if np.any( self.As) == None:
            raise ValueError("Bath parameters must first be set with set_bath_params before running simulation")
            
        self.ou_verlet(self.system, self.forcefield, nt, self.dt, self.kbT, 
                       self.As, self.Bs, self.Avs, self.Asv, PBC = self.PBC, 
                       reporters = self.reporters, reportints = self.reportints)
        pass     

    @staticmethod
    def ou_verlet(system, forcefield, nt, dt, kbT, As, Bs, Avs, Asv, PBC = False, reporters = [], reportints = []):
        """
        Generalized Langevin Dynamics with Verlet hopping algorithm. 
        Uses independent baths for each paricle in system.
    
        K(t) = k0. Avs . exp(-As t) . Asv ~ scalar
        k0 ~ Bs Bs.T / (2 As + As.T) (when As and Bs commute)
        
        Parameters
        ----------
        system : System Object.
        forcefield : Forcefield Object.
        As : Numpy Array.
            Ornstein-Uhlenbeck Drift Matrix
        Bs : Numpy Array.
            Ornstein-Uhlenbeck Random Multiplier Matrix
        Asv : Numpy Array.
            System -> Auxiliary projection Matrix
        Asv : Numpy Array.
            Auxiliary ->  System projection Matrix
        kbT : Float.
            Thermal Energy.
        nt : Int
            Number of timesteps.
        dt : Float
            Length of each timestep.
            
        reporters : list of reporter objects, optional
        reportints : list of reporter intervals, optional
        
        Returns
        -------
        reporters : list of reporter objects
            Data saved from simulation in specified reporter objects.
        """
        
        # Number of atoms/dimensions
        nsys  = system.nsys
        ndim  = system.ndim
        naux  = np.size(Bs,axis=0)
        
        #mass/forcefield
        m = system.m
        
        # Create Position Array
        x_t = system.pos
        
        # Create Velocity Array
        v_t = system.vel
        
        # Create Auxiliary Variable Arrays
        s_t = np.zeros((nsys,naux,ndim))

            
        # Create Force Arrays
        ff = forcefield
        f_t = ff.calc_frc(x_t)
            
        # Create Total Force Arrays
        a_t = np.zeros((nsys,ndim))
        
        # Save Initial State
        for k in range(len(reportints)):
            reporters[k].save(x_t,v_t,a_t,f_t)
            
        # Run Simulation
        print("Running Simulation")
        for i in range(1,nt):
            # Move velocity half-step
            a_t_half = f_t/m + np.einsum("fj,njd->nd",Avs,s_t)/m
            v_t_half = v_t + dt/2.0 * a_t_half
            
            # Move Position full-step
            x_t = x_t + dt * v_t_half
            if PBC:
                x_t = PBCwrap(x_t, system.box_l)
            
            # Calculate Force at new position
            f_t = ff.calc_frc(x_t)
            
            # Move auxiliary variables full-step
            s_selfpart = -np.einsum("ij,njd->nid", As, s_t)
        
            s_syspart  = -np.einsum("if,nd->nid", Asv, v_t_half)
            
            noise = np.random.normal(loc=0.0, scale=1.0, size=(nsys,naux,ndim))
            s_ranpart =  np.einsum("ij,njd->nid",Bs,noise)
            
            s_t = s_t + (dt * s_selfpart) + (dt * s_syspart) + (np.sqrt(dt) * s_ranpart)
    
            # Move velocity full-step
            a_t = f_t/m + np.einsum("fj,njd->nd",Avs,s_t)/m
            v_t = v_t_half + dt/2.0 * a_t
            
            # Report
            for k in range(len(reportints)):
                if i % reportints[k]  == 0:
                    reporters[k].save(x_t,v_t,a_t,f_t)
        
        # Save Simulation State to System Object
        system.pos = x_t.copy()
        system.vel = v_t.copy()
        
        pass

class gld_corr(object):
    """ 
    Class for integrating the Generalized Langevin Dynamics.
    """
    
    def __init__(self,system, forcefield, dt, style=1, PBC = False, reporters = [], reportints = []):
        """
        Parameters
        ----------
        system : System Object.
        forcefield : Forcefield Object.
        nt : Int.
            Number of timesteps.
        dt : Float.
            Length of each timestep.d
        
        Optional
        ----------
        style : String. 
            Style option for integrator, can be one of: ou, ou-corr, 
            or ou-sprs-corr.
        PBC : Bool. 
            Whether or not to use periodic boundary conditions, Default=False
        s0 : Float or Numpy Array. 
            Initial positions of bath particles. Default is 0
        reporters : list of reporter objects
        reportints : list of reporter intervals

        """
        
        #Set General Integration Parameters
        self.system = system
        self.forcefield = forcefield
        self.dt = dt
        self.style = int(style)
        self.PBC = PBC
        self.reporters = reporters
        self.reportints = reportints
        
        #Check valid Choice of style
        if self.style not in ["ou", "ou-corr", "ou-corr-sprs"]:
            ValueError(" Style must be one of: verlet, verlet-corr, or verlet-sprs-cor ")
        
        #Set Ornstein-Uhlenbeck Back Parameters
        self.As = None
        self.Bs = None
        self.Asv = None
        self.Avs = None
        self.kbT = 0
        pass
        
    def set_bath_params(self,As, Bs, Asv, Avs, temp, k0=None):
        """
        Set Bath Parameters.

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
            Ornstein-Uhlenbeck stationary variance. Default set to identity
        """
        
        # Assign Parameters
        self.As = As
        self.Bs = Bs
        self.Asv = Asv
        self.Avs = Avs
        self.kbT = kb * temp
        
        # Check that dimensions match
        nsys  = self.system.nsys
        naux  = np.size(Bs,axis=0)
        assert naux == np.size(Bs,axis=1) == np.size(As,axis=0) == np.size(As,axis=1)
        assert naux == np.size(Avs,axis=1) == np.size(Asv,axis=0)
        assert nsys == np.size(Avs,axis=0) == np.size(Asv,axis=1)
            
        # Assign stationary variance
        # If k0 is not assigned, assume it is identity or already folded into Asv
        if np.all(k0) != None:
            self.Asv = k0 @ self.Asv
    
        pass
            
        
    def run(self, nt):
        if np.any([self.As, self.Bs, self.Asv, self.Avs]) == None:
            raise ValueError("Bath parameters must first be set with set_bath_params before running simulation")
            
        if self.style == 1:
            self.reporters = self.ou_corr(self.system, self.forcefield, nt, self.dt, 
                                       self.kbT, self.As, self.Bs, self.Avs, 
                                       self.Asv, PBC = self.PBC,
                                       reporters = self.reporters, reportints = self.reportints)
        elif self.style == 2:
            self.reporters = self.ou_corr_sprs(self.system, self.forcefield, nt, self.dt, 
                                       self.kbT, self.As, self.Bs, self.Avs, 
                                       self.Asv, PBC = self.PBC,
                                       reporters = self.reporters, reportints = self.reportints)
        else:
            raise ValueError ("Style must be one of (1) - Verlet, (2) - Verlet with Sparse Linear Algebra")
            
        pass

    def ou_corr(system, forcefield, nt, dt, kbT, As, Bs, Avs, Asv, PBC = False, reporters = [], reportints = []):
        """
        Generalized Langevin Dynamics with Verlet hopping algorithm. 
        Uses global bath parameters instead of per-particle bath parameters.
    
        K(t) = Avs . k0 . exp(-As t) . Asv ~ NxN matrix
        k0 ~ Bs Bs.T / (2 As + As.T) (when As and Bs commute)
        
        Parameters are the same as ou_verlet
        """
        
        # Number of atoms/dimensions
        nsys  = system.nsys
        ndim  = system.ndim
        naux  = np.size(Bs,axis=0)
        
        #mass/forcefield
        m = system.m
        
        # Create Position Array
        x_t = system.pos
        
        # Create Velocity Array
        v_t = system.vel
        
        # Create Auxiliary Variable Arrays
        s_t = np.zeros((naux,ndim))
            
        # Create Force Arrays
        ff = forcefield
        f_t = ff.calc_frc(x_t)
            
        # Create Total Force Arrays
        a_t = np.zeros((nsys,ndim))
        
        # Save Initial State
        for k in range(len(reportints)):
            reporters[k].save(x_t,v_t,a_t,f_t)
            
        # Run Simulation
        print("Running Simulation")
        for i in range(1,nt):
            
            # Move velocity half-step
            a_t_half = f_t/m + np.einsum("ij,jd->id",Avs,s_t)/m
            v_t_half = v_t + dt/2.0 * a_t_half
            
            # Move Position full-step
            x_t = x_t + dt * v_t_half
            if PBC:
                x_t = PBCwrap(x_t, system.box_l)
            
            
            # Calculate Force at new position
            f_t = ff.calc_frc(x_t)
            
            # Move auxiliary variables full-step
            s_selfpart = -np.einsum("ij,jd->id", As, s_t)
        
            s_syspart  = -np.einsum("ij,jd->id", Asv, v_t_half)
            
            noise = np.random.normal(loc=0.0, scale = 1.0, size=(naux,ndim))
            s_ranpart =  np.einsum("ij,jd->id",Bs,noise)
            
            s_t = s_t + (dt * s_selfpart) + (dt * s_syspart) + (np.sqrt(dt) * s_ranpart)
    
            # Move velocity full-step
            a_t = f_t/m + np.einsum("ij,jd->id",Avs,s_t)/m
            v_t = v_t_half + dt/2.0 * a_t
            
            # Report
            for k in range(len(reportints)):
                if i % reportints[k]  == 0:
                    reporters[k].save(x_t,v_t,a_t,f_t)
        
            # Save Simulation State to System Object
            system.pos = x_t
            system.vel = v_t
            
        return reporters

    def ou_corr_sprs(system, forcefield, nt, dt, kbT, As, Bs, Avs, Asv, k0, PBC=False, reporters = [], reportints = []):
        """
        Generalized Langevin Dynamics with Verlet hopping algorithm. 
        Uses global bath parameters instead of per-particle bath parameters.
        Uses sparse linear algebra libraries for matrix multiplication
    
        K(t) = Avs . k0 . exp(-As t) . Asv
        k0 ~ Bs Bs.T / (2 As + As.T) (when As and Bs commute)
        
        Parameters are the same as ou_verlet
        """
        
        # Number of atoms/dimensions
        nsys  = system.nsys
        ndim  = system.ndim
        naux  = np.size(Bs,axis=0)
            
        k0_sprs  = sp.bsr_matrix(k0)
        As_sprs  = sp.bsr_matrix(As)
        Bs_sprs  = sp.bsr_matrix(Bs)
        Avs_sprs = sp.bsr_matrix(Avs)
        Asv_sprs = sp.bsr_matrix(Asv)
        
        #mass
        m = system.m
        
        # Create Position Array
        x_t = system.pos
        
        # Create Velocity Array
        v_t = system.vel
        
        # Create Auxiliary Variable Arrays
        s_t = np.zeros((naux,ndim))

        # Create Force Arrays
        ff = forcefield
        f_t = ff.calc_frc(x_t)
            
        # Create Total Force Arrays
        a_t = np.zeros((nsys,ndim))
        
        # Save Initial State
        for k in range(len(reportints)):
            reporters[k].save(x_t,v_t,a_t,f_t)
            
        # Run Simulation
        print("Running Simulation")
        for i in range(1,nt):
            
            # Move velocity half-step
            a_t_half = f_t/m + Avs_sprs.dot(s_t)/m
            v_t_half = v_t + dt/2.0 * a_t_half
            
            # Move Position full-step
            x_t = x_t + dt * v_t_half
            if PBC:
                x_t = PBCwrap(x_t, system.box_l)
            
            # Calculate Force at new position
            f_t = ff.calc_frc(x_t)
            
            # Move auxiliary variables full-step
            s_selfpart =  -As_sprs.dot(s_t)
        
            s_syspart  =  -k0_sprs.dot(Asv_sprs).dot(v_t_half)
            
            noise = np.random.normal(loc=0.0, scale=1.0, size=(naux,ndim))
            s_ranpart  =  Bs_sprs.dot(noise)
            
            s_t = s_t + (dt * s_selfpart) + (dt * s_syspart) + (np.sqrt(dt) * s_ranpart)
    
            # Move velocity full-step
            a_t = f_t/m + Avs_sprs.dot(s_t)/m
            v_t = v_t_half + dt/2.0 * a_t
            
            # Report
            for k in range(len(reportints)):
                if i % reportints[k]  == 0:
                    reporters[k].save(x_t,v_t,a_t,f_t)
        
        # Save Simulation State to System Object
        system.pos = x_t
        system.vel = v_t
        
        return reporters

if __name__ == "__main__":
    pass