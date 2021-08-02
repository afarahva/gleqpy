#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyGLE
============

Functions for extracting time-correlation functions and memory kernels using
Numpy arrays.
"""

import numpy as np

def calc_tcf(obs1,obs2,max_t,stride=1):
    """
    Compute time correlation function between two observables up to a max_t

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
        Time Correlation Function.

    """
    nt = np.size(obs1,axis=0)
    natom = np.size(obs1,axis=1)

    tcf = np.zeros((max_t//stride,natom),dtype=np.float64)
    i = 0
    for t in range(0,max_t,stride):
        tcf_t = 1.0/(nt - t) * np.einsum("tjd,tjd->j",obs1[t:],obs2[0:nt-t])
        tcf[i,:] = tcf_t
        i += 1

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

def calc_memory_fft(vel_tcf, frc_tcf):
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

    return memory

def calc_memory_midpt(vel_tcf,frc_tcf,dt, K_0=None):
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
    max_t = np.size(vel_tcf,axis=0)
    natom = np.size(vel_tcf,axis=1)

    # Midpoints of velocity autocorrelation function
    vel_tcf_mid = 0.5* (vel_tcf[1:] + vel_tcf[0:-1])
    
    # Calc and store V(0.5)^(-1)
    vel_tcf_mid0_inv = np.linalg.inv(vel_tcf_mid[0])

    ##### Iteratively invert convolution operator in real-time
    K = np.zeros((max_t-1,natom,natom))
    
    # Assign initial condition
    if np.all(K_0) != None:
        K[0] = K_0
    else:
        K[0] = np.dot( - frc_tcf[1]/dt, vel_tcf_mid0_inv)
        
    # Loop through remaining indices
    for t in range(1, max_t -1):
        flip_v = np.flip(vel_tcf_mid[1:t+1],axis=0)
        
        temp = - frc_tcf[t+1]/dt - np.einsum("tij,tjk->ik", K[0:t] , flip_v)
              
        K_t = np.linalg.solve( vel_tcf_mid[0].T, temp.T).T
        #np.dot(temp, np.linalg.inv(vel_tcf_mid[0])
           
        K[t] = K_t
    
    return K

def calc_memory_dtrapz(d_vel_tcf, d_frc_tcf, vel_tcf_0, dt,K_0=None):
    """
    Calculate memory kernel in real-time using derivative formula and 
    trapezoidal quadrature.
    
    Parameters
    ----------
    d_vel_tcf: Numpy Array.
        Time derivative of velocity time correlation function
    d_frc_tcf: Numpy Array.
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
    max_t = np.size(d_vel_tcf,axis=0)
    natom = np.size(d_vel_tcf,axis=1)
    
    # Calc and store V(0)^(-1)
    vel_inv0 = np.linalg.inv(vel_tcf_0)
    vel_invt = np.linalg.inv(vel_tcf_0 + dt/2 * d_vel_tcf[0])
    
    ##### Iteratively invert convolution operator in real-time
    K = np.zeros((max_t-1,natom,natom))
    
    # Set initial Values
    if np.all(K_0) != None:
        K[0] = K_0
    else:
        K[0] = np.dot( -d_frc_tcf[0], vel_inv0 )
    
    # Loop through remaining indices
    for t in range(1, max_t-1):
            
        flip_v = np.flip(d_vel_tcf[1:t+1],axis=0)
        
        temp1 = np.einsum("ij,jk->ik", K[0] , flip_v[0])
        
        if t==1:
            temp2 = 0
        else:
            temp2 = 2 * np.einsum("tij,tjk->ik", K[1:t] , flip_v[1:])
        
        temp = -d_frc_tcf[t] - dt/2 * (temp1 + temp2)
        
        K_t = np.dot(temp, vel_invt)
           
        K[t] = K_t
    
    return K

if __name__ == "__main__":
    pass