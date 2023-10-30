# -*- coding: utf-8 -*-
"""
GLEPy
============

file: time.py
description: Functions for extracting time-correlation functions and memory 
kernels using numpy arrays.
"""

import numpy as np
from scipy.signal import correlate as spcorrelate
from scipy.optimize import curve_fit


########## Tools for time-correlation functions and memory kernel extraction 
########## from MD snapshots

def calc_tcf(obs1, obs2, max_t, stride=1, mode="scipy"):
    """
    Compute time correlation function between two observables up to a max_t

    Parameters
    ----------
    obs1 : Numpy Array.
        First Observable. (nt,nDoF)
    obs2 : Numpy Array.
        Second Observable. (nt,nDoF)
    max_t : Int
        Maximum time to compute time-correlation function.
    mode: Str.
        How to calculate time-correlation function. 
        Default "fft" or "scipy" (faster).
        Other option is "direct" for real time method. 

    Returns
    -------
    tcf : Numpy Array.
        Time Correlation Function.

    """
    
    nt = np.size(obs1,axis=0)
    nDoF = np.size(obs1,axis=1)

    tcf = np.zeros((max_t//stride,nDoF),dtype=np.float64)
    
    #use direct method
    if mode == "direct":
        i = 0
        for t in range(0,max_t,stride):
            tcf_t = 1.0/(nt - t) * np.sum(obs1[t:] * obs2[0:nt-t], axis=0)
            tcf[i,:] = tcf_t
            i += 1
            
    elif mode == "fft" or mode=="scipy":
        for i in range(nDoF):
            corr = spcorrelate(obs1[:,i],obs2[:,i],mode="same")[nt//2 : nt//2 + max_t : stride]
            tcf[:,i] = corr/np.arange(nt,nt-max_t,-1)
            
            
    else:
        raise ValueError("mode must be either direct or fft")

    return tcf

def calc_matrix_tcf(obs1,obs2,max_t,stride=1):
    """
    Compute matrix time correlation function between two vector observbables.

    Parameters
    ----------
    obs1 : Numpy Array.
        First Observable. (nt,natom,3)
    obs2 : Numpy Array.
        Second Observable. (nt,natom,3)
    max_t : Int
        Maximum time to compute tcf.

    Returns
    -------
    tcf : Numpy Array.
        Time Correlation Function. (nt,natom,natom)

    """
    nt = np.size(obs1,axis=0)
    natom = np.size(obs1,axis=1)

    tcf = np.zeros((max_t//stride,natom,natom))
    i = 0
    for t in range(0,max_t,stride):
        tcf_t = 1.0/(nt - t) * np.einsum("tik,tjk->ij",obs1[t:],obs2[0:nt - t])
        tcf[i,:,:] = tcf_t
        i += 1
    return tcf

def calc_memory_fft(vel_tcf, frc_tcf, dt):
    """
    Calculate memory kernel using fourier transform and convolution thm.
    
    <F(t),v(0)> = -\int K(t)<v(t),v(0)>
    K(\omega) = -C_F(\omega) * C_V^{-1}(\omega)
    K(t) = ifft(K(\omega))

    Parameters
    ----------
    vel_tcf: Numpy Array. (Nt)
        Velocity auto-correlation function. <v(t),v(0)>
    frc_tcf: Numpy Array. (Nt)
        Force time correlation function. <F(t),v(0)>
    dt: Float.
        Time window spacing (must be in same units as vel_tcf and frc_tcf)
        
    Returns
    -------
    memory : Numpy Array.
        Memory Kernel. K(t)
    """
    if len(vel_tcf.shape) > 1 or len(frc_tcf.shape) > 1:
        raise ValueError("TCF input must be 1D Numpy Array")
        
    #Calculate fourier transforms of velocity/force autocorrelation functions
    vel_matrix_fft = np.fft.fft(vel_tcf)
    frc_matrix_fft = np.fft.fft(frc_tcf)
    
    #Calculate memory kernel
    memory = np.fft.ifft(-frc_matrix_fft/vel_matrix_fft,axis=0)
    
    return memory/dt

def calc_memory_midpt(vel_tcf, frc_tcf, dt, K_0=None):
    """
    Calculate memory kernel in real-time using midpoint quadrature.
    
    K(t-0.5) = 1/V(0.5) * [ -F(t)/dt 
        - \sum_{\tau = 0.5}^{t-1.5} V(t - \tau)K(\tau) ]

    Parameters
    ----------
    vel_tcf: Numpy Array. (Nt)
        Velocity time correlation function
    frc_tcf: Numpy Array. (Nt)
        Force time correlation function
    dt: Float.
        Time window spacing (must be in same units as vel_tcf and frc_tcf)
    K_0: Numpy Array. Optional.
        Initial value of memory kernel.

    Returns
    -------
    memory : Numpy Array.
        Memory Kernel. K(t)
    """
    if len(vel_tcf.shape) > 1 or len(frc_tcf.shape) > 1:
        raise ValueError("TCF input must be 1D Numpy Array")
        
    ##### Parameters
    Nt = np.size(vel_tcf)

    # Midpoints of velocity autocorrelation function
    vel_tcf_mid = 0.5* (vel_tcf[1:] + vel_tcf[0:-1])
    
    # Calc and store V(0.5)^(-1)
    vel_tcf_inv0 = 1.0/(vel_tcf_mid[0])

    ##### Iteratively invert convolution operator in real-time
    K = np.zeros(Nt-1,dtype=np.float64)
    
    # Assign initial condition
    if np.all(K_0) != None:
        K[0] = K_0
    else:
        K[0] = - frc_tcf[1] * vel_tcf_inv0 / dt
        
    # Loop through remaining indices
    for t in range(1, Nt-1):
        flip_v = np.flip(vel_tcf_mid[1:t+1],axis=0)
        
        temp = - frc_tcf[t+1]/dt - np.sum( K[0:t] *  flip_v )
              
        K_t = vel_tcf_inv0 * temp
           
        K[t] = K_t
    
    return K

def calc_memory_dtrapz(dvel_tcf, dfrc_tcf, vel_tcf_0, dt, K_0=None):
    """
    Calculate memory kernel in real-time using derivative formula and 
    trapezoidal quadrature.
    
    Parameters
    ----------
    dvel_tcf: Numpy Array. (Nt)
        Time derivative of velocity time correlation function
    dfrc_tcf: Numpy Array. (Nt)
        Time derivative of force time correlation function
    vel_tcf_0: Float.
        Mean-square velocity.
    dt: Float.
        Time window spacing (must be in same units as vel_tcf and frc_tcf)
    K_0: Numpy Array. Optional.
        Initial value of memory kernel.

    Returns
    -------
    memory : Numpy Array.
        Memory Kernel. K(t)
    """
    
    if len(dvel_tcf.shape) > 1 or len(dfrc_tcf.shape) > 1:
        raise ValueError("TCF input must be a 1D Numpy Array")
        
    ##### Parameters
    Nt = np.size(dvel_tcf)
    
    # Calc and store V(0)^(-1)
    vel_inv0 = 1 / vel_tcf_0
    vel_invt = 1/ (vel_tcf_0 + dt/2 * dvel_tcf[0])
    
    ##### Iteratively invert convolution operator in real-time
    K = np.zeros(Nt,dtype=np.float64)
    
    # Set initial Values
    if np.all(K_0) != None:
        K[0] = K_0
    else:
        K[0] = -dfrc_tcf[0] * vel_inv0
    
    # Loop through remaining indices
    for t in range(1, Nt):
            
        flip_v = np.flip(dvel_tcf[1:t+1],axis=0)
        
        temp1 = K[0] * flip_v[0]
        
        if t==1:
            temp2 = 0
        else:
            temp2 = 2 * np.sum( K[1:t] * flip_v[1:] )
        
        temp = -dfrc_tcf[t] - dt/2 * (temp1 + temp2)
        
        K_t = vel_invt * temp
           
        K[t] = K_t
    
    return K

def calc_matrix_memory_fft(vel_tcf, frc_tcf, dt):
    """
    Calculate memory kernel using fourier transform and convolution thm.
    
    <F(t),v(0)> = -\int K(t)<v(t),v(0)>
    K(\omega) = -C_F(\omega) * C_V^{-1}(\omega)
    K(t) = ifft(K(\omega))

    Parameters
    ----------
    vel_tcf: Numpy Array.
        Velocity auto-correlation function. <v(t),v(0)>
    frc_tcf: Numpy Array.
        Force time correlation function. <F(t),v(0)>

    Returns
    -------
    memory : Numpy Array.
        Memory Kernel. K(t)
    """

    #Calculate fourier transforms of velocity/force autocorrelation functions
    vel_matrix_fft = np.fft.fft(vel_tcf,axis=0)
    frc_matrix_fft = np.fft.fft(frc_tcf,axis=0)

    #Calculate memory kernel
    memory_ft = np.linalg.solve(vel_matrix_fft,-frc_matrix_fft)
    memory = np.fft.ifft(memory_ft,axis=0)

    return memory/dt

def calc_matrix_memory_midpt(vel_tcf, frc_tcf, dt, K_0=None):
    """
    Calculate memory kernel in real-time using midpoint quadrature.
    
    K(t-0.5) = 1/V(0.5) * [ -F(t)/dt 
        - \sum_{\tau = 0.5}^{t-1.5} V(t - \tau)K(\tau) ]

    Parameters
    ----------
    vel_tcf: Numpy Array.
        Velocity time correlation function
    frc_tcf: Numpy Array.
        Force time correlation function
    dt: Float.
        Time window spacing (must be in same units as vel_tcf and frc_tcf)
    K_0: Numpy Array. Optional.
        Initial value of memory kernel.

    Returns
    -------
    K : Numpy Array.
        Memory Kernel.
    """
    ##### Parameters
    Nt = np.size(vel_tcf,axis=0)
    natom = np.size(vel_tcf,axis=1)

    # Midpoints of velocity autocorrelation function
    vel_tcf_mid = 0.5* (vel_tcf[1:] + vel_tcf[0:-1])
    
    # Calc and store V(0.5)^(-1)
    vel_tcf_mid0_inv = np.linalg.inv(vel_tcf_mid[0])

    ##### Iteratively invert convolution operator in real-time
    K = np.zeros((Nt,natom,natom))
    
    # Assign initial condition
    if np.all(K_0) != None:
        K[0] = K_0
    else:
        K[0] = np.dot( - frc_tcf[1]/dt, vel_tcf_mid0_inv)
        
    # Loop through remaining indices
    for t in range(1, Nt):
        flip_v = np.flip(vel_tcf_mid[1:t+1],axis=0)
        
        temp = - frc_tcf[t+1]/dt - np.einsum("tij,tjk->ik", K[0:t] , flip_v)
              
        K_t = np.linalg.solve( vel_tcf_mid[0].T, temp.T).T
           
        K[t] = K_t
    
    return K

def calc_matrix_memory_dtrapz(dvel_tcf, dfrc_tcf, vel_tcf_0, dt, K_0=None):
    """
    Calculate memory kernel in real-time using derivative formula and 
    trapezoidal quadrature.
    
    Parameters
    ----------
    dvel_tcf: Numpy Array.
        Time derivative of velocity time correlation function
    dfrc_tcf: Numpy Array.
        Time derivative of force time correlation function
    vel_tcf_0: Numpy Array.
        Mean-square velocity.
    dt: Float.
        Time window spacing (must be in same units as vel_tcf and frc_tcf)
    K_0: Numpy Array. Optional.
        Initial value of memory kernel.

    Returns
    -------
    K : Numpy Array.
        Memory Kernel.
    """
    ##### Parameters
    Nt = np.size(dvel_tcf,axis=0)
    natom = np.size(dvel_tcf,axis=1)
    
    # Calc and store V(0)^(-1)
    vel_inv0 = np.linalg.inv(vel_tcf_0)
    vel_invt = np.linalg.inv(vel_tcf_0 + dt/2 * dvel_tcf[0])
    
    ##### Iteratively invert convolution operator in real-time
    K = np.zeros((Nt,natom,natom))
    
    # Set initial Values
    if np.all(K_0) != None:
        K[0] = K_0
    else:
        K[0] = np.dot( -dfrc_tcf[0], vel_inv0 )
    
    # Loop through remaining indices
    for t in range(1, Nt):
            
        flip_v = np.flip(dvel_tcf[1:t+1],axis=0)
        
        temp1 = np.einsum("ij,jk->ik", K[0] , flip_v[0])
        
        if t==1:
            temp2 = 0
        else:
            temp2 = 2 * np.einsum("tij,tjk->ik", K[1:t] , flip_v[1:])
        
        temp = -dfrc_tcf[t] - dt/2 * (temp1 + temp2)
        
        K_t = np.dot(temp, vel_invt)
           
        K[t] = K_t
    
    return K

########## Useful matrix functions

def matrixexp(mat, t,  Hermitian=True, return_eig=False):
    """
    Calculate Time-Dependent Matrix Exponential of a General Matrix

    Parameters
    ----------
    mat : Numpy Array. (ndim,dim)
        Matrix.
    t : Numpy Array. (nt)
        Timesteps.
    mode: String. OPTIONAL
        Specifies whether matrix is matrix is Hermitian or not. Default is "H".
    return_eig: Bool. OPTIONAL
        Specifies whether to return eigenvalues/vectors. 

    Returns
    -------
    mexp : Numpy Array. (nt,ndim,ndim)
        Matrix Exponential.
    """
    assert np.size(mat,axis=0) == np.size(mat,axis=1)

    nt = np.size(t)
    ndim = np.size(mat,axis=0)
    indx = np.arange(ndim)
    
    if Hermitian:
        eige, eigv = np.linalg.eigh(mat)
    else:
        eige, eigv = np.linalg.eig(mat)

    diag = np.zeros((nt, ndim,ndim),dtype=np.complex128)
    diag[:,indx,indx] = np.exp( np.outer(t,eige) )
                  
    if Hermitian:
        mexp = eigv @ diag @ eigv.T
    else:
        mexp = eigv @ diag @ np.linalg.inv(eigv)
    
    if return_eig:
        return mexp, eige, eigv
    else:
        return mexp
    

def matrixexp_deriv(mat, t,  Hermitian=True, return_eig=False):
    """
    Calculate Time-Dependent Matrix Exponential of a General Matrix

    Parameters
    ----------
    mat : Numpy Array. (ndim,dim)
        Matrix.
    t : Numpy Array. (nt)
        Timesteps.
    mode: String. OPTIONAL
        Specifies whether matrix is matrix is Hermitian or not. Default is "H".
    return_eig: Bool. OPTIONAL
        Specifies whether to return eigenvalues/vectors. 

    Returns
    -------
    mexp : Numpy Array. (nt,ndim,ndim)
        Matrix Exponential.
    """
    assert np.size(mat,axis=0) == np.size(mat,axis=1)

    nt = np.size(t)
    ndim = np.size(mat,axis=0)
    indx = np.arange(ndim)
    
    if Hermitian:
        eige, eigv = np.linalg.eigh(mat)
    else:
        eige, eigv = np.linalg.eig(mat)

    diag = np.zeros((nt, ndim,ndim),dtype=np.complex128)
    diag[:,indx,indx] = np.exp( np.outer(t,eige) )/eige
                  
    if Hermitian:
        mexp = eigv @ diag @ eigv.T
    else:
        mexp = eigv @ diag @ np.linalg.inv(eigv)
    
    if return_eig:
        return mexp, eige, eigv
    else:
        return mexp

def matrix_transform(left, center, right):
    """
    Triple Matrix Product, M(t) = L @ C(t) @ R

    Parameters
    ----------
    left : Numpy Array (Ni,Nj)
        Left Matrix.
    center : Numpy Array. (Nt, Nj, Nk)
        Center Matrix, Time-Dependent
    right : TYPE
        Right Matrix.

    Returns
    -------
    product : Numpy Array. (Nt, Ni, Nk)
        Final product.
    """
    Nt = np.size(center,axis=0)
    product = np.zeros((Nt,left.shape[0],right.shape[1]),dtype=np.complex128)
    for t in range(Nt):
        product[t,:,:] = left @ center[t] @ right
    return product

def matrixfunc(func, mat, t, Hermitian=True, return_eig=False):
    """
    Calculate Time-Dependent Matrix Function

    Parameters
    ----------
    mat : Numpy Array. (ndim,dim)
        Matrix.
    t : Numpy Array. (nt)
        Timesteps.
    mode: String. OPTIONAL
        Specifies whether matrix is matrix is Hermitian or not. Default is "H".
    return_eig: Bool. OPTIONAL
        Specifies whether to return eigenvalues/vectors. 

    Returns
    -------
    mfunc : Numpy Array. (nt,ndim,ndim)
        Matrix Function.
    """
    assert np.size(mat,axis=0) == np.size(mat,axis=1)
    nt = np.size(t)
    ndim = np.size(mat,axis=0)
    indx = np.arange(ndim)
    
    if Hermitian:
        eige, eigv = np.linalg.eigh(mat)
    else:
        eige, eigv = np.linalg.eig(mat)
        
    diag = np.zeros((nt, ndim,ndim),dtype=np.complex128)
    diag[:,indx,indx] = func( np.outer(t,eige) )
    
    if Hermitian:
        mfunc = eigv @ diag @ eigv.T
    else:
        mfunc = eigv @ diag @ np.linalg.inv(eigv)

    if return_eig:
        return mfunc, eige, eigv
    else:
        return mfunc


def matrix_cos(modes, freq, t, Hermitian=True):
    """
    Calculate Time-Dependent Cosine Function of a matrix given associated modes.
    and frequencies

    Parameters
    ----------
    modes : Numpy Array. (nmodes,nmodes)
        Eigenmodes.
    freq : Numpy Array. (nmodes)
        Eigenfrequencies
    t : Numpy Array. (nt)
        Timesteps.
    mode: String. OPTIONAL
        Specifies whether matrix is matrix is Hermitian or not. Default is "H".

    Returns
    -------
    M : Numpy Array. (nt,ndim,ndim)
        Matrix Sin.
    """
    Ndof = np.size(freq)
    Nt = np.size(t)
    
    indx = np.arange(Ndof)
    M = np.zeros((Nt,Ndof,Ndof),dtype=np.float64)
    M[:,indx,indx] = np.cos( np.outer(t, freq ) ) 
    if Hermitian:
        M = modes @ M @ modes.T
    else:
        M = modes @ M @ np.linalg.inv(modes)
    return M

def matrix_sin(modes, freq, t, Hermitian=True):
    """
    Calculate Time-Dependent Sine Function of a matrix given associated modes. 
    and frequencies

    Parameters
    ----------
    modes : Numpy Array. (nmodes,nmodes)
        Eigenmodes.
    freq : Numpy Array. (nmodes)
        Eigenfrequencies
    t : Numpy Array. (nt)
        Timesteps.
    mode: String. OPTIONAL
        Specifies whether matrix is matrix is Hermitian or not. Default is "H".
        
    Returns
    -------
    M : Numpy Array. (nt,ndim,ndim)
        Matrix Sin.
    """
    Ndof = np.size(freq)
    Nt = np.size(t)
    
    indx = np.arange(Ndof)
    M = np.zeros((Nt,Ndof,Ndof),dtype=np.float64)
    M[:,indx,indx] = np.sin( np.outer(t, freq ) ) 
    if Hermitian:
        M = modes @ M @ modes.T
    else:
        M = modes @ M @ np.linalg.inv(modes)
    return M

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

########## Useful operations for fitting memory kernels to A-matrices

def calc_spectral(signals, scale):
    """
    Calculate positive fast-fourier transform of a real, even signal.
    Scales output to the appropriate height. 
    """
    output = []
    for i in range(len(signals)):
        f_i = np.abs( np.real( np.fft.fft(signals[i]) ) ) /scale
        output.append( f_i[0:len(f_i)//2-1])
    return output


class Amatrix(object):
    """
    Class used to fit memory kernels to a sum of exponentially damped cosines
    and output parameters of the GLE A-matrix. 
    
    K =  \sum_i C_i * e^(-\gamma t) * cos(\omega_i*t)
      = A_{ps} @ e^{-t A_s} A_{sp}
           
    A_sp =  \sqrt{C/2}, \sqrt{C/2}
    A_sp = -\sqrt{C/2}, \sqrt{C/2}
    A_s  = [[2*\gamma, sqrt(\gamma^2 + \omega^2)],[-sqrt(\gamma^2 + \omega^2), 0]]
    """
    
    def __init__(self, coeffs=None,decays=None,freqs=None,Amat=None):
        """
        Initialize variables
        """
        
        
        self.coeffs = coeffs # coefficients of each term (C_i)
        self.decays = decays # expoential decay factors (\gamma_i)
        self.freqs = freqs   # frequencies (\omega_i)
        self.Amat = Amat     # GLE A matrix
        
    #--------------------------------------
    ##### converting and writing A matrices
    def params_to_Amat(self):
        """
        Converts damped cosine parameters to GLE A-matrix
    
        Requires class variables "coeffs", "decays", and "freqs" to be defined
            
        Returns
        -------
        A_mat : Numpy Array,
            GLE A Matrix.
        """
        if self.coeffs is None or self.decays is None or self.freqs is None:
            raise ValueError("class variables 'coeffs', 'decays', and 'freqs' must be defined")
            
        assert len(self.coeffs) == len(self.decays) == len(self.freqs)
        
        N = len(self.coeffs)
        self.Amat = np.zeros((2*N+1, 2*N+1),dtype=np.float64)
        
        for i in range(1,2*N+1,2):
            n = (i-1)//2
            C,gamma,omega = self.coeffs[n], self.decays[n], self.freqs[n]
            
            self.Amat[i,i] = 2*gamma
            self.Amat[i,i+1] = -np.sqrt(gamma**2 + omega**2)
            self.Amat[i+1,i] = np.sqrt(gamma**2 + omega**2)
            
            self.Amat[0,i:i+2] = -np.sqrt(C/2)
            self.Amat[i:i+2,0] = np.sqrt(C/2)
        
        return self.Amat

    def Amat_to_params(self):
        """
        Converts GLE A-matrix to  damped cosine parameters 
        
        K =  \sum_i a_i * e^(-d t) * cos(w*t)
        
        Requires class variable "Amat" to be defined
    
        Returns
        -------
        coeffs : Numpy Array.
            Coefficients of exponential sinusoid decay.
        decays :  Numpy Array.
            Decay Rates.
        freqs : Numpy Array.
            Decay Frequencies.
    
        """
        if self.Amat is None:
            raise ValueError("class variables 'Amat' must be defined")
            
        N = (np.size(self.Amat,axis=0)-1)//2
        
        self.coeffs, self.decays, self.omegas = np.zeros(N), np.zeros(N), np.zeros(N)
        for i in range(N):
            n = 2*i
            self.coeffs[i]  = 2*self.Amat[0,n+1]**2
            self.decays[i] = self.Amat[n+1,n+1]/2
            self.omegas[i] = np.sqrt(self.Amat[n+1,n+2]**2 - self.decays[i]**2)
            
        return self.coeffs, self.decays, self.omegas
    
    def calc_friction(self):
        """
        Calculate Markovian friciton constant.
        
        Returns
        -------
        friction : Float.
            Total Markovian friction
        friction_arr :  Numpy Array.
            Friction per term in memory kernel
        """
        
        if self.coeffs is None or self.decays is None or self.freqs is None:
            try:
                self.Amat_to_params()
            except:
                raise ValueError("no GLE parameters found")
            
        N = len(self.coeffs)
        friction = 0
        friction_arr = []
        for i in range(N):
            fr_i = self.coeffs[i] * self.decays[i]/(self.decays[i]**2+self.omegas[i]**2)
            friction_arr.append(fr_i)
            friction+=fr_i
        return  friction, friction_arr
    
    def write_Amatrix(self, file):
        """
        write A matrix to file to be read by LAMMPS GLE.
    
        Parameters
        ----------
        file : String.
            Filename.
        """
        if self.Amat is None:
            try:
                self.params_to_Amat()
            except:
                raise ValueError("no GLE parameters found")
            
        fout = open(file,"w")
        for i in range( np.size(self.Amat,axis=1) ):
            line = np.array2string(self.Amat[i], separator=" ", max_line_width=np.inf, formatter={'float_kind':lambda x: "%.7f" % x})[1:-1]
            fout.write(" " + line + "\n")
        fout.close()
        pass

    #--------------------------------------
    ##### fitting memory kernel
    @staticmethod
    def multiterm_lorentz(nterm):
        """
        Wrapper which returns a function that takes N+1 arguments, where N is
        the number of terms in a sum of Lorentzians
        """
        
        def lorentz(coeff, decay, omega_0, w):
            """
            Lorentzian Function
            """
            f = coeff * ( decay/(decay**2 + (w - omega_0)**2) )
            return f

        
        def f_output(w_arr,*args):
            coeffs = args[0:nterm]
            decays = args[nterm:2*nterm]
            omegas = args[2*nterm:3*nterm]
            result = 0
            for i in range(nterm):
                result = result + lorentz(coeffs[i],decays[i],omegas[i],w_arr)
            return result
        
        return f_output
    
    @staticmethod
    def multiterm_exp(nterm):
        """
        Wrapper which returns a function that takes N+1 arguments, where N is
        the number of terms in a sum of damped exponentials
        """
        
        def damped_exp(coeff, decay, omega_0, t):
            """
            Damped Cos-Exponential Function
            """
            f = coeff * np.cos(omega_0*t) * np.exp(-decay*t)
            return f
        
        def f_output(t_arr,*args):
            coeffs = args[0:nterm]
            decays = args[nterm:2*nterm]
            omegas = args[2*nterm:3*nterm]
            result = 0
            for i in range(nterm):
                result = result + damped_exp(coeffs[i],decays[i],omegas[i],t_arr)
            return result
        
        return f_output
    
    @staticmethod
    def arrays_to_input(coeffs,decays,omegas):
        """
        Convert seperate parameters arrays to single input to function
        """
        input_args = []
        input_args.extend(coeffs)
        input_args.extend(decays)
        input_args.extend(omegas)
        return input_args
    
    @staticmethod
    def input_to_arrays(input_args, nterm):
        """
        Convert single input to funciton to seperate parameters arrays
        """
        coeffs = input_args[0:nterm]
        decays = input_args[nterm:2*nterm]
        omegas = input_args[2*nterm:3*nterm]
        return coeffs,decays,omegas
    
    def fit(self,t_arr, Kt, nterm, 
                    coeffs_guess, decays_guess, freqs_guess, **kwargs):
        """
        Fit memory kernel(Kt) to a sum of exponentially damped cosines

        Parameters
        ----------
        t_arr : Numpy array.
            time axis.
        Kt : Numpy array
            memory kernel (or any time-dependent function).
        nterm : Int.
            Number of terms.
        coeffs_guess : Numpy array.
            Guesses for coefficient of each term.
        decays_guess : Numpy array.
            Guesses for exponential decays of each term.
        freqs_guess : Numpy array.
            Guesses for frequencies of each term.
            
        **kwargs 
        
        Returns
        -------
        Kt_fit : Numpy array
            fitted memory kernel (or any time-dependent function).
        coeffs_fit : Numpy array.
            Fits for coefficient of each term.
        decays_fit : Numpy array.
            Fits for exponential decays of each term.
        freqs_fit : Numpy array.
            Fits for frequencies of each term.
        """
        
        guess = self.arrays_to_input(coeffs_guess,decays_guess,freqs_guess)
        func  = self.multiterm_exp(nterm)
        popt, pcov = curve_fit(func, t_arr, Kt, p0=guess, **kwargs)
        
        # Scale Coefficients such that K(t=0) == K_{fit}(t=0)
        popt[0:nterm] = popt[0:nterm] * Kt[0]/np.sum(popt[0:nterm])

        # Expand Parameters to calculate fitted kernel
        Kt_fit = func(t_arr,*popt)
        coeffs_fit, decays_fit, freqs_fit = self.input_to_arrays(popt,nterm)
        
        # Set class parameters
        self.coeffs = coeffs_fit 
        self.decays = decays_fit 
        self.freqs  = freqs_fit  
        
        return Kt_fit, coeffs_fit, decays_fit, freqs_fit

if __name__ == "__main__":
    pass