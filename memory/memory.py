#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyGLE
============

Functions for extracting time-correlation functions and memory kernels using
Numpy arrays.
"""

import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.signal import correlate as spcorrelate

def calc_tcf(obs1, obs2, max_t, stride=1, mode="direct"):
    """
    Compute time correlation function between two observables up to a max_t

    Parameters
    ----------
    obs1 : Numpy Array.
        First Observable. (nt,nDoF)
    obs2 : Numpy Array.
        Second Observable. (nt,nDoF)
    max_t : Int
        Maximum time to compute tcf.

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

def prony(signal, dt, Nterm):
    """
    Prony's method, fit's a signal to a series of complex exponentials.
    F(t) = \sum_i A_i * exp(B_i t)

    Parameters
    ----------
    signal : Numpy Array.
        1D Array containing signal at evenly sampled points.
    dt : Float.
        Sampling interval.
    Nterm :
        Number of terms complex exponential terms desired

    Returns
    -------
    A : Numpy Array.
        Coefficients.
    B : Numpy Array.
        Exponents.
    """

	# Create Matrices to Solve for Polynomial Coefficients
    # v = F.dot(coeff)
    Nt = np.size(signal)
    Bmat = np.zeros((Nt-Nterm, Nterm),dtype=np.complex128)
    v = signal[Nterm:Nt]

	# Solve for polynomial coefficients using least squares
    for i in range(Nterm):
        Bmat[:, i] = signal[Nterm-i-1:Nt-i-1]
		
    coeff = np.linalg.lstsq(Bmat, v,rcond=None)[0]
    coeffP = np.append(-np.flip(coeff),1.0)
    
	# Find roots of characteristic polynomial, these are the exponents. 
    roots = poly.polyroots(coeffP)
    B = np.log(roots)/(dt)

	# Find prefactors of exponential. 
    Amat = np.zeros((Nt, Nterm),dtype=np.complex128)

    for i in range(Nt):
        Amat[i, :] = roots**i

    
    A = np.linalg.lstsq(Amat, signal, rcond=None)[0]
    return A, B

def prony_real(A, B, eps=1e-10):
    """
    Converts complex Prony series to a real series.
    F(t) = \sum_i A_i * exp(B_i t)
         = \sum_i a_i exp(c_i t) cos( d_i t) + b_i exp(c_i t) sin( d_i t)

    Parameters
    ----------
    A : Numpy Array.
        Coefficients.
    B : Numpy Array.
        Exponents.

    Returns
    -------
    coeff_cos : Numpy Array.
    coeff_sin : Numpy Array.
    decay : Numpy Array.
    freq : Numpy Array.
    """
    indx = np.where(np.abs(np.imag(A)) < eps)
    re_A = A[indx]
    re_B = B[indx]
    A = np.delete(A,indx)
    B = np.delete(B,indx)
    
    # check consistency
    assert np.size(A) == np.size(B)
    
    # 
    coeff_cos = np.real(A[0:-1:2] + A[1::2])
    coeff_sin = -np.imag(A[0:-1:2] - A[1::2])
    decay = np.real(B[0:-1:2])
    freq = np.imag(B[0:-1:2])
    
    coeff_cos = np.append(coeff_cos, np.real(re_A))
    coeff_sin = np.append(coeff_sin, np.zeros(np.size(re_A)))
    decay = np.append(decay, np.real(re_B))
    freq = np.append(freq, np.zeros(np.size(re_B)))
    
    return coeff_cos, coeff_sin, decay, freq

def pronysum_complex(A,B,tarr):
    """
    Reconstruct signal by summing Prony Series
    
    Prony's method, fit's a signal to a series of complex exponentials.
    F(t) = \sum_i A_i * exp(B_i t)

    Parameters
    ----------
    A : Numpy Array.
        Coefficients.
    B : Numpy Array.
        Exponents.
    tarr : Numpy Array.
        Time array.

    Returns
    -------
    signal : Numpy Array.
        Output Signal.
    """
    
    Nterm = np.size(A)
    signal = 0
    for i in range(Nterm):
        signal += A[i] * np.exp(B[i]*tarr)
    return signal

def pronysum_real(a,b,c,d,tarr):
    """
    F(t) = \sum_i a_i exp(c_i t) cos( d_i t) + b_i exp(c_i t) sin( d_i t)

    Parameters
    ----------
    a : Numpy Array.
        cosine coefficients.
    b : Numpy Array.
        sine coefficients.
    c : Numpy Array.
        exponential decay rates.
    d : Numpy Array.
        oscillation frequencies.
    tarr : Numpy Array.
        Time array.

    Returns
    -------
    signal : Numpy Array.
        Output Signal.
    """

    Nterm = np.size(a)
    signal = 0
    for i in range(Nterm):
        signal += np.exp(c[i] * tarr) * (a[i]*np.cos(d[i]*tarr) + b[i]*np.sin(d[i]*tarr))
    return signal

if __name__ == "__main__":
    pass