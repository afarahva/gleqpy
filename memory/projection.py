import numpy as np

class BathProjection(object):
    """
    Calculates memory kernel, spectral density, and bath averaged potential
    constant, by projecting away bath dynamics in a purely Harmonic system.
    
    Methods
    ----------
    1) calc_memory: Memory Kernel
    2) calc_spectral: Spectral Density
    3) calc_springk: PMF spring constant
    """
    
    def __init__(self, hess, masses, indx_P):
        """
        Projects mass-weighed Hessian into system and bath subspaces.
        
        d^x/dt^2 = D * x
        
        Parameters
        ----------
        hess : Numpy Array. (nDoF,nDoF)
            Hessian Matrix.
        masses : Numpy Array. (nDoF)
            Per Atom Masses.
        indx_P : Numpy Array. 
            Indices of system degrees of freedom
        """
            
        # Number of system and bath degrees of freedom
        N = np.size(hess,axis=0)
        self.indx_P = indx_P
        self.indx_Q = np.delete(np.arange(N),self.indx_P)
        self.nsys = len(self.indx_P)
        self.nbath = len(self.indx_Q)
        
        # Mass Array
        if type(masses) != np.ndarray:
            self.masses = np.ones(3*N) * masses
        elif len(masses) != N:
            self.masses = np.repeat(masses,3)
        else:
            self.masses = masses
        
        isqrtm = self.masses**-0.5
        
        # Mass weighted Hessian
        self.D = isqrtm[:, None] * hess * isqrtm[None,:]
        
        self.D_PP = self.D[np.ix_(self.indx_P,self.indx_P)]
        self.D_PQ = self.D[np.ix_(self.indx_P,self.indx_Q)]
        self.D_QP = self.D[np.ix_(self.indx_Q,self.indx_P)]
        self.D_QQ = self.D[np.ix_(self.indx_Q,self.indx_Q)]
        
        # Diagonalize Bath Hessian
        if np.all(np.isclose(self.D_QQ,self.D_QQ.T)):
            self.diagonalize_symm() # uses symmetric diagonalization
        else:
            self.diagonalize() # uses general technique
        
        pass
    
    def diagonalize_symm(self):
        """
        Diagonalize Bath Hessian and calculate related matrices. 
        Assumes Bath Hessian is a symmetric matrix
        """
        
        # Calculate Bath modes/frequencies
        freq2, modes = np.linalg.eigh(self.D_QQ)
        freq2 = np.where(freq2 > 0, freq2, 0)
        
        self.freq2 = freq2.copy()
        self.freq = np.sqrt(freq2)
        self.ifreq2 = np.where(freq2 == 0, 0, 1/freq2)
        self.ifreq = np.sqrt(self.ifreq2)
        
        self.modes = modes.copy()
        
        # Calculate coupling matrices
        self.iC = self.D_PQ @ self.modes
        self.C = self.modes.T @ self.D_QP
        
        # Calculate Inverse Eigenvalue Diagonal Matrix
        self.iW2 = np.diag(self.ifreq2)

        pass

    def diagonalize(self):
        """
        Diagonalize Bath Hessian and calculate related matrices. 
        """
        
        # Calculate Bath modes/frequencies
        freq2, modes = np.linalg.eig(self.D_QQ)
        argsort = np.argsort(freq2)
        freq2 = freq2[argsort]
        modes = modes[:,argsort]
        freq2 = np.where(freq2 > 0, freq2, 0)
        
        self.freq2 = freq2.copy()
        self.freq = np.sqrt(freq2)
        self.ifreq2 = np.where(freq2 == 0, 0, 1/freq2)
        self.ifreq = np.sqrt(self.ifreq2)
        
        self.modes = modes.copy()
        
        # Calculate coupling matrices
        self.iC = self.D_PQ @ self.modes
        self.C = np.linalg.inv(self.modes) @ self.D_QP
        
        # Calculate Inverse Eigenvalue Diagonal Matrix
        self.iW2 = np.diag(self.ifreq2)
        pass
        

    def calc_memory(self,t_arr):
        """
        Calculates Memory Kernel. 

        Parameters
        ----------
        t_arr : Numpy Array. (Nt)
            Time axis.

        Returns
        -------
        K : Numpy Array. (Nt,Nsys,Nsys)
            Memory Kernel.
        """
        
        # Calculate bath-mode cos matrix - cos(W*t)/W^2
        Nt = len(t_arr)
        indx = np.arange(self.nbath)
        self.cosM = np.zeros((Nt,self.nbath,self.nbath),dtype=np.float64)
        self.cosM[:,indx,indx] = np.cos( np.outer(t_arr,self.freq) )*self.ifreq2
        
        # Transform through system-bath coupling to memory kernel
        # D_{PQ} @ U.T @ cos(Wt)/W^2 @ U @ D_QP
        sqrtm = self.masses[self.indx_P]**0.5
        
        self.K = (sqrtm[None,:,None] * 
                  self.iC @ self.cosM @ self.C * 
                  sqrtm[None,None,:])
        
        # Transform from mass-weighted to standard coordinates
        diagim = np.diag( 1/self.masses[self.indx_P] )
        
        K = diagim @ self.K  
        
        return K
    
    def calc_spectraldensity(self):
        """
        Calculate J(\omega) = C.T @ 1/W^2 @ C

        Returns
        -------
        None.

        """
        self.J = np.einsum("aj,j,jb->jab",self.iC,1/(self.freq**2),self.C)
        return self.J 
        
    def calc_springk(self):
        """
        Calculates spring constant k for potential of mean force

        U = (x - mu).T @ k/2 @ (x-mu)
        
        Returns
        -------
        k : Numpy Array. (3,3)
            Spring Constant Matrix.
        """
        
        # Calculate bath-mode inverse frequency matrix
        sqrtm = self.masses[self.indx_P]**0.5
        
        # Transform through system bath coupling
        self.Omega2 = self.iC @ self.iW2 @ self.C
        
        # Transform from mass-weighted to standard coordinates
        self.k_s = sqrtm[:, None] * self.D_PP * sqrtm[None,:]
        self.k_b = sqrtm[:, None] * self.Omega2 * sqrtm[None,:]
        
        # Total Hooke's law matrices
        self.k_eff = (self.k_s - self.k_b)
        return self.k_eff
    
class BathProjectionCustom(object):
    """
    Calculates memory kernel, spectral density, and bath averaged potential
    constant, by projecting away bath dynamics in a purely Harmonic system.
    
    Uses user specified P and Q operators.
    
    Methods
    ----------
    1) calc_memory: Memory Kernel
    2) calc_spectral: Spectral Density
    3) calc_springk: PMF spring constant
    """
    
    def __init__(self, hess, masses, indx_sys, P, Q):
        """
        Projects mass-weighed Hessian into system and bath subspaces.
        
        d^x/dt^2 = D * x
        P ~ System Projection Operator
        Q ~ Bath Projection Operator
        
        Parameters
        ----------
        hess : Numpy Array. (nDoF,nDoF)
            Hessian Matrix.
        masses : Numpy Array. (natom)
            Per Atom Masses.
        P : Numpy Array. 
            System projection operator. 
        Q : Numpy Array. 
            Bath projection operator.
        """
            
        # Calculate Mass-Weighted Hessian from true Hessian
        self.masses = masses
        im = np.repeat(masses**-0.5, 3)
        self.D = im[:, None] * hess * im
        self.nsys = np.size(P,axis=0)
        self.nbath = np.size(Q,axis=0)

        # Determine if projection operators are square or rectangular
        if self.nsys == self.nbath == hess.shape[0]:
            self.mode = "square"
        elif self.nsys != self.nbath and self.nsys == len(indx_sys):
            self.mode = "rectangular"
        else:
            raise ValueError("""P and Q must either be square matrices, 
                    or rectangullar matrices with the proper dimensions""")
            
        # Split into system/bath blocks appropriately. 
        if self.mode == "rectangular":
            self.indx_sys = indx_sys
            self.D_PP = P @ self.D @ P.T
            self.D_QQ = Q @ self.D @ Q.T
            self.D_QP = Q @ self.D @ P.T
            self.D_PQ = P @ self.D @ Q.T
            
        else:
            self.indx_sys = np.arange(self.nsys)
            self.D_PP = P @ self.D @ P
            self.D_QQ = Q @ self.D @ Q
            self.D_QP = Q @ self.D @ P
            self.D_PQ = P @ self.D @ Q
        
        
        # Diagonalize Bath Hessian
        if np.all(np.isclose(self.D_QQ,self.D_QQ.T)):
            self.diagonalize_symm() # uses symmetric diagonalization
        else:
            self.diagonalize() # uses general technique
        
        pass
    
    def diagonalize_symm(self):
        """
        Diagonalize Bath Hessian and calculate related matrices. 
        Assumes Bath Hessian is a symmetric matrix
        """
        
        # Calculate Bath modes/frequencies
        freq2, modes = np.linalg.eigh(self.D_QQ)
        freq2 = np.where(freq2 > 0, freq2, 0)
        
        self.freq2 = freq2.copy()
        self.freq = np.sqrt(freq2)
        self.ifreq2 = np.where(freq2 == 0, 0, 1/freq2)
        self.ifreq = np.sqrt(self.ifreq2)
        
        self.modes = modes.copy()
        
        # Calculate coupling matrices
        self.iC = self.D_PQ @ self.modes
        self.C = self.modes.T @ self.D_QP
        
        # Calculate Inverse Eigenvalue Diagonal Matrix
        self.iW2 = np.diag(self.ifreq2)

        pass

    def diagonalize(self):
        """
        Diagonalize Bath Hessian and calculate related matrices. 
        """
        
        # Calculate Bath modes/frequencies
        freq2, modes = np.linalg.eig(self.D_QQ)
        argsort = np.argsort(freq2)
        freq2 = freq2[argsort]
        modes = modes[:,argsort]
        freq2 = np.where(freq2 > 0, freq2, 0)
        
        self.freq2 = freq2.copy()
        self.freq = np.sqrt(freq2)
        self.ifreq2 = np.where(freq2 == 0, 0, 1/freq2)
        self.ifreq = np.sqrt(self.ifreq2)
        
        self.modes = modes.copy()
        
        # Calculate coupling matrices
        self.iC = self.D_PQ @ self.modes
        self.C = np.linalg.inv(self.modes) @ self.D_QP
        
        # Calculate Inverse Eigenvalue Diagonal Matrix
        self.iW2 = np.diag(self.ifreq2)
        pass
        

    def calc_memory(self,t_arr):
        """
        Calculates Memory Kernel. 

        Parameters
        ----------
        t_arr : Numpy Array. (Nt)
            Time axis.

        Returns
        -------
        K : Numpy Array. (Nt,Nsys,Nsys)
            Memory Kernel.
        """
        
        # Calculate bath-mode cos matrix - cos(W*t)/W^2
        Nt = len(t_arr)
        indx = np.arange(self.nbath)
        self.cosM = np.zeros((Nt,self.nbath,self.nbath),dtype=np.float64)
        self.cosM[:,indx,indx] = np.cos( np.outer(t_arr,self.freq) )*self.ifreq2
        
        # Transform through system-bath coupling to memory kernel
        # D_{PQ} @ U.T @ cos(Wt)/W^2 @ U @ D_QP
        sqrtm = np.repeat(self.masses**0.5, 3)[self.indx_sys]
        self.K = (sqrtm[None,:,None] * 
                  self.iC @ self.cosM @ self.C * 
                  sqrtm[None,None,:])
        
        # Transform from mass-weighted to standard coordinates
        diagim = np.diag( np.repeat(1/self.masses, 3)[self.indx_sys] )
        
        K = diagim @ self.K  
        
        return K
    
    def calc_spectraldensity(self):
        """
        Calculate J(\omega) = C.T @ 1/W^2 @ C

        Returns
        -------
        None.

        """
        self.J = np.einsum("aj,j,jb->jab",self.iC,1/(self.freq**2),self.C)
        return self.J 
        
    def calc_springk(self):
        """
        Calculates spring constant k for potential of mean force

        U = (x - mu).T @ k/2 @ (x-mu)
        
        Returns
        -------
        k : Numpy Array. (3,3)
            Spring Constant Matrix.
        """
        
        # Calculate bath-mode inverse frequency matrix
        sqrtm   = np.repeat(self.masses**0.5, 3)[self.indx_sys]
        
        # Transform through system bath coupling
        self.Omega2 = self.iC @ self.iW2 @ self.C
        
        # Transform from mass-weighted to standard coordinates
        self.k_s = sqrtm[:, None] * self.D_PP * sqrtm[None,:]
        self.k_b = sqrtm[:, None] * self.Omega2 * sqrtm[None,:]
        
        # Total Hooke's law matrices
        self.k_eff = (self.k_s - self.k_b)
        return self.k_eff

    
def proj_orthogonal(N, indx_P):
    """
    Construct orthogonal projection operators.

    Parameters
    ----------
    N : Int.
        Number of degrees of freedom.
        
    indx_P : Numpy Array.
        Indices of 'system' degrees of freedom.

    Returns
    -------
    P : Numpy Array. 
        System projection operator. 
    
    Q : Numpy Array. 
        Bath projection operator.
    """
    
    Nsys = len(indx_P)
    Nbath = N - Nsys
    indx_Q = np.delete(np.arange(N),indx_P)
    
    # Rectangular system projection Operator
    P = np.zeros( ( Nsys , N) )
    P[ np.arange(Nsys), indx_P ] = 1 
    
    # Rectangular bath projection operator
    Q = np.zeros( ( Nbath, N) )
    Q[ np.arange(Nbath), indx_Q ] = 1

    return P, Q

def proj_orthogonal_square(N, indx_P):
    """
    Construct square orthogonal projection operators.

    Parameters
    ----------
    N : Int.
        Number of degrees of freedom.
        
    indx_P : Numpy Array.
        Indices of 'system' degrees of freedom.

    Returns
    -------
    P : Numpy Array. 
        System projection operator. 
    
    Q : Numpy Array. 
        Bath projection operator.
    """
    
    # System projection operator
    P = np.zeros( ( N , N) )
    P[ indx_P,indx_P ] = 1 
    
    # Bath projection operator
    Q = np.eye(N) - P


    return P, Q

def proj_mori(N, indx_P, cov):
    """
    Construct orthogonal projection operators.

    Parameters
    ----------
    N : Int.
        Number of degrees of freedom.
        
    indx_P : Numpy Array.
        Indices of 'system' degrees of freedom.

    Returns
    -------
    P : Numpy Array. 
        System projection operator. 
    
    Q : Numpy Array. 
        Bath projection operator.
    """
    
    indx_Q = np.delete(np.arange(N),indx_P)

    P = np.zeros((N,N))
    
    P[np.ix_(indx_P,indx_P)] = np.eye(len(indx_P)) #cov[ np.ix_(indx_P,indx_P) ]/cov[indx_P,indx_P]
    P[np.ix_(indx_Q,indx_P)] = cov[ np.ix_(indx_Q,indx_P) ]/cov[indx_P,indx_P]
    Q = np.eye(N) - P
    
    return P, Q

# def proj_zwanzig(N, indx_P, H, masses):
#     """
#     Construct Zwanzig's projection operator.

#     Parameters
#     ----------
#     N : Int.
#         Number of degrees of freedom.
        
#     indx_P : Numpy Array.
#         Indices of 'system' degrees of freedom.

#     Returns
#     -------
#     P : Numpy Array. 
#         System projection operator. 
    
#     Q : Numpy Array. 
#         Bath projection operator.
#     """
    
#     Po, Qo = proj_orthogonal(P, indx_P)
#     bath = BathProjection(H, masses, indx_P, P, Q)
    
#     freq2, mods, C = bath.freq2, bath.modes, bath.C
    
#     Pz = np.zeros((N,N))
#     for i in range(len(indx_P)):
#         a = 2
#     zwanzig = -np.einsum("ij,j->i", modes, (C + iC)/(2*freq2))
    
#     return P, Q



# def calc_memory_from_npz(path, dt, stride):
#     npz = np.load(path)
#     vel_tcf, dvel_tcf, frc_tcf, dfrc_tcf = npz["vel_tcf"], npz["dvel_tcf"], npz["frc_tcf"], npz["dfrc_tcf"]
    
#     vel_tcf_ave = vel_tcf.mean(axis=1)
#     vel_tcf_x   = vel_tcf[:,0:-2:3].mean(axis=-1)
#     vel_tcf_y   = vel_tcf[:,1:-1:3].mean(axis=-1)
#     vel_tcf_z   = vel_tcf[:,2::3].mean(axis=-1)
    
#     dvel_tcf_ave = dvel_tcf.mean(axis=1)
#     dvel_tcf_x   = dvel_tcf[:,0:-2:3].mean(axis=-1)
#     dvel_tcf_y   = dvel_tcf[:,1:-1:3].mean(axis=-1)
#     dvel_tcf_z   = dvel_tcf[:,2::3].mean(axis=-1)
    
#     dfrc_tcf_ave = dfrc_tcf.mean(axis=1)
#     dfrc_tcf_x   = dfrc_tcf[:,0:-2:3].mean(axis=-1)
#     dfrc_tcf_y   = dfrc_tcf[:,1:-1:3].mean(axis=-1)
#     dfrc_tcf_z   = dfrc_tcf[:,2::3].mean(axis=-1)
    
#     Kt_x = mem.calc_memory_dtrapz(dvel_tcf_x, dfrc_tcf_x, vel_tcf_x[0], dt*stride)
#     Kt_y = mem.calc_memory_dtrapz(dvel_tcf_y, dfrc_tcf_y, vel_tcf_y[0], dt*stride)
#     Kt_z = mem.calc_memory_dtrapz(dvel_tcf_z, dfrc_tcf_z, vel_tcf_z[0], dt*stride)
#     Kt_ave = mem.calc_memory_dtrapz(dvel_tcf_ave, dfrc_tcf_ave, vel_tcf_ave[0], dt*stride)
#     return Kt_x, Kt_y, Kt_z, Kt_ave


if __name__ == "__main__":
    pass