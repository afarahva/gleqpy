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
    
    eige, eigv = np.linalg.eig(mat)
    
    diag = np.zeros((ndim,ndim,nt),dtype=np.complex128)
    diag[np.diag_indices(ndim)[0],np.diag_indices(ndim)[1],:] = np.exp( np.outer(eige,t) )
                   
    mexp = np.einsum("ij,jkt,km->tim",eigv,diag,np.linalg.inv(eigv))
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
    
    eige, eigv = np.linalg.eigh(mat)
    
    diag = np.zeros((ndim,ndim,nt),dtype=np.complex128)
    diag[np.diag_indices(ndim)[0],np.diag_indices(ndim)[1],:] = np.exp( np.outer(eige,t) )
                   
    mexp = np.einsum("ij,jkt,km->tim",eigv,diag,eigv.T)
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
    
    def __init__(self,nsys,ndim,box_l):
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

        self.m = np.ones((nsys,1))
        
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
        v_sq = np.sum(v**2,axis=1)
        temp = np.mean( (0.5 * self.m * v_sq)/(0.5 * self.ndim * self.kb) )
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
    
    #If operators A, A.T, B, and B.T don't commute
    if method == "full":
            A_mexp = matrixexp(-A,t_arr)
            B_sq = B.dot(B.T)
            integrand = np.einsum("tij,jk,tlk->til",A_mexp, B_sq, A_mexp)
            var0 = np.trapz(integrand, t_arr, axis=0)
            statvar = np.real( np.einsum("ij,jk,kl->il",Avs, var0, Asv) )
        
    #If operators A, A.T, B, and B.T do commute
    elif method == "commuting":
            B_sq = B.dot(B.T)
            A_inv = np.linalg.solve(A + A.T, np.identity(naux))
            var0  = np.einsum("ij,jk->ik", A_inv, B_sq)
            statvar = np.einsum("ij,jk,kl->il",Avs, var0, Asv)
        
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
        reporters = self.verlet(self.system, self.forcefield, self.nt, self.dt, 
                           PBC = self.PBC, reporters = self.reporters, 
                           reportints = self.reportints)
        return reporters
    
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
                PBCwrap(x_t, system.box_l)
            
            # Calculate Force at new position
            f_t = ff.calc_frc(x_t)
    
            # Move velocity full-step
            v_t = v_t_half + (dt/2.0 * f_t)/m
                            
            # Report
            for k in range(len(reportints)):
                if i % reportints[k]  == 0:
                    reporters[k].save(x_t,v_t,a_t,f_t)
        
        # Save Simulation State to System Object
        system.pos = x_t
        system.vel = v_t
        
        return reporters        
    
class langevin(object):
    """
    Class for integrating langevin dynamics.
    """
    def  __init__(self, system, forcefield, dt, PBC=False, reporters = [], reportints = []):
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
        
        #Set bath Parameters
        self.A = None
        self.B = None
        self.kbT = None
        pass
    
    def set_bath_params(self, kbT, A, B=None):
        """
        Set Bath Parameters.

        Parameters
        ----------
        kbT : Float.
            Thermal Energy
        A : Float or 1D/2D Numpy Array.
            Friction Constant/Array.
        B : Numpy Array. (Optional)
            Wiener Multiplier Array (Default set to obey Fluctuation-Dissipation Thm.)
        """
        self.kbT = kbT
        nsys  = self.system.nsys
        # ndim  = self.system.ndim
        
        # Check Array Types
        self.A = A
        if type(A) != np.ndarray:
            self.A = np.ones(nsys) * A
        
        # Check dimensions and assign random force array
        if self.A.ndim == 2:
            assert self.system.nsys == np.size(self.A, axis=0) == np.size(self.A, axis=1)

            if np.all(B) == None:
                self.B = np.sqrt(self.kbT) * np.linalg.cholesky(self.A + self.A.T)
            
        elif self.A.ndim <= 2:
            assert self.system.nsys == np.size(self.A)

            if np.all(B) == None:
                self.B = np.sqrt(2 * kbT * self.A)
                
        else:
            raise ValueError("A must either be a float, or a 1/2D Numpy Array")
        pass
        

    def run(self, nt):
        """
        Run Calculations for nt steps.
        """
        if self.A.ndim == 2:
            self.reporters = self.verlet_corr(self.system, self.forcefield, nt, self.dt, 
                                    self.kbT, self.A, self.B, PBC = self.PBC, 
                                    reporters = self.reporters, reportints = self.reportints)
           
        else:  
            self.reporters = self.verlet(self.system, self.forcefield, nt, self.dt, 
                                    self.kbT, self.A[:,None], self.B[:,None], PBC = self.PBC, 
                                    reporters = self.reporters, reportints = self.reportints)
        return self.reporters
        
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
            noise = np.random.normal(loc=0.0, scale=(1/np.sqrt(ndim)), size=(nsys,ndim))
            stoch = B * noise
            det = f_t - A * v_t
            
            v_t_half = v_t + (dt/2.0 * det)/m + (np.sqrt(dt)/2.0 * stoch)/m
                        
            # Move Position full-step
            x_t = x_t + dt * v_t_half
            if PBC:
                PBCwrap(x_t, system.box_l)
            
            # Calculate Force at new position
            f_t = ff.calc_frc(x_t)
    
            # Move velocity full-step
            det = f_t - A * v_t_half
            a_t = det/m + stoch/m
            v_t = v_t_half + (dt/2.0 * det)/m + (np.sqrt(dt)/2.0 * stoch)/m
                            
            # Report
            for k in range(len(reportints)):
                if i % reportints[k]  == 0:
                    reporters[k].save(x_t,v_t,a_t,f_t)
        
        # Save Simulation State to System Object
        system.pos = x_t
        system.vel = v_t
        
        return reporters
    
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
            
            v_t_half = v_t + (dt/2.0 * det)/m + (np.sqrt(dt)/2.0 * stoch)/m
            
            # Move Position full-step
            x_t = x_t + dt * v_t_half
            if PBC:
                PBCwrap(x_t, system.box_l)
            
            # Calculate Force at new position
            f_t = ff.calc_frc(x_t)
    
            # Move velocity full-step
            det = f_t - A.dot(v_t_half)
            a_t = det/m + stoch/m
            v_t = v_t_half + (dt/2.0 * det)/m + (np.sqrt(dt)/2.0 * stoch)/m
                            
            # Report
            for k in range(len(reportints)):
                if i % reportints[k]  == 0:
                    reporters[k].save(x_t,v_t,a_t,f_t)
        
        # Save Simulation State to System Object
        system.pos = x_t
        system.vel = v_t
        
        return reporters

class generalized_langevin(object):
    """ 
    Class for integrating the Generalized Langevin Dynamics.
    """
    
    def __init__(self,system, forcefield, dt, style="ou", PBC = False, s0 = None, reporters = [], reportints = []):
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
        self.s0 = s0
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
        self.k0 = None
        pass
        
    def set_bath_params(self,As, Bs, Asv, Avs, kbT, k0=None):
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
        kbT : Float.
            Thermal Energy.
        k0 : Numpy Array. 
            Ornstein-Uhlenbeck stationary variance. Default set to identity
        """
        # Assign Parameters that dimensions match

        self.As = As
        self.Bs = Bs
        self.Asv = Asv
        self.Avs = Avs
        self.kbT = kbT
        
        # Check that dimensions match
        nsys  = self.system.nsys
        # ndim  = self.system.ndim
        naux  = np.size(Bs,axis=0)
        
        if self.style == "ou":
            assert naux == np.size(Bs,axis=1) == np.size(As,axis=0) == np.size(As,axis=1)
            assert naux == np.size(Avs,axis=1) == np.size(Asv,axis=0)
            assert 1 == np.size(Avs,axis=0) == np.size(Asv,axis=1)
            
        elif self.style in ["ou-corr", "ou-corr-sprs"]:
            assert naux == np.size(Bs,axis=1) == np.size(As,axis=0) == np.size(As,axis=1)
            assert naux == np.size(Avs,axis=1) == np.size(Asv,axis=0)
            assert nsys == np.size(Avs,axis=0) == np.size(Asv,axis=1)
            
        # Assign stationary variance
        if np.all(k0) == None:
            self.k0 = np.identity(naux)
        elif type(k0) == str:
            self.k0 = calc_ou_var(Bs, As, 10000, self.dt, method=k0)/(kbT)
        else:
            self.k0 = k0
        pass
            
        
    def run(self, nt):
        if self.style == "ou":
            self.reporters = self.ou_verlet(self.system, self.forcefield, nt, self.dt, 
                                       self.kbT, self.As, self.Bs, self.Avs, 
                                       self.Asv, self.k0, PBC = self.PBC, s0 = self.s0,
                                       reporters = self.reporters, reportints = self.reportints)
        elif self.style == "ou-corr":
            self.reporters = self.ou_corr(self.system, self.forcefield, nt, self.dt, 
                                       self.kbT, self.As, self.Bs, self.Avs, 
                                       self.Asv, self.k0, PBC = self.PBC, s0 = self.s0,
                                       reporters = self.reporters, reportints = self.reportints)
        elif self.style == "ou-corr-sprs":
            self.reporters = self.ou_corr_sprs(self.system, self.forcefield, nt, self.dt, 
                                       self.kbT, self.As, self.Bs, self.Avs, 
                                       self.Asv, self.k0, PBC = self.PBC, s0 = self.s0,
                                       reporters = self.reporters, reportints = self.reportints)
        return self.reporters
        
    @staticmethod
    def ou_verlet(system, forcefield, nt, dt, kbT, As, Bs, Avs, Asv, k0, PBC = False, s0 = None, reporters = [], reportints = []):
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
        if np.any(s0) == None:
            s_t = np.zeros((nsys,naux,ndim))
        else:
            s_t = s0
            
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
            s_selfpart =  np.einsum("ij,njd->nid", -As, s_t)
        
            s_syspart  = np.einsum("ij,jf,nd->nid", -k0, Asv, v_t_half)
            
            noise = np.random.normal(loc=0.0, scale=(1.0/np.sqrt(ndim)), size=(nsys,naux,ndim))
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
        system.pos = x_t
        system.vel = v_t
        
        return reporters

    def ou_corr(system, forcefield, nt, dt, kbT, As, Bs, Avs, Asv, k0, PBC = False, s0 = None, reporters = [], reportints = []):
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
        if s0 == None:
            s_t = np.zeros((naux,ndim))
        else:
            s_t = s0
            
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
            s_selfpart =  np.einsum("ij,jd->id", -As, s_t)
        
            s_syspart  = np.einsum("ij,jk,kd->id", -k0, Asv, v_t_half)
            
            noise = np.random.normal(loc=0.0, scale=(1.0/np.sqrt(ndim)), size=(naux,ndim))
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

    def ou_corr_sprs(system, forcefield, nt, dt, kbT, As, Bs, Avs, Asv, k0, PBC=False, s0 = None, reporters = [], reportints = []):
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
        if s0 == None:
            s_t = np.zeros((naux,ndim))
        else:
            s_t = s0
            
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
            
            noise = np.random.normal(loc=0.0, scale=(1.0/np.sqrt(ndim)), size=(naux,ndim))
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
