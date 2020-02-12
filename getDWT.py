import pywt
import numpy as np

def getDWT(X,N,Name):

    eps = np.finfo(float).eps
    
    # Preform wavelet decomposition
    [cA1,cD1] = pywt.dwt(X,Name)
    [cA2,cD2] = pywt.dwt(cA1,Name)
    [cA3,cD3] = pywt.dwt(cA2,Name)

    # Reorder the details based on the structure of the wavelet
    # decomposition
    d1 = cD1
    d1 = np.vstack((d1,d1))
    d1 = d1.reshape(-1,1,order='F')
    d2 = cD2
    d2 = np.vstack((d2,d2,d2,d2))
    d2 = d2.reshape(-1,1,order='F')
    d3 = cD3
    d3 = np.vstack((d3,d3,d3,d3,d3,d3,d3,d3))
    d3 = d3.reshape(-1,1,order='F')

    cD_1 = centered_slice(d1,len(X))
    cD_2 = centered_slice(d2,len(X))
    cD_3 = centered_slice(d3,len(X))
    cD = np.concatenate((cD_1,cD_2,cD_3),axis=1)
    cD = cD.reshape(cD.size,1,order='C')

    # Space cD according to spacing of floating point numbers:
    ind = np.argwhere(abs(cD.ravel())<np.sqrt(eps))
    cD[ind] = 0
    cD = cD.reshape(N,X.size,order='F')

    # Reorder the approximations based on the structure of the wavelet
    # decomposition
    a1 = cA1
    a1 = np.vstack((a1,a1))
    a1 = a1.reshape(-1,1,order='F')
    a2 = cA2
    a2 = np.vstack((a2,a2,a2,a2))
    a2 = d2.reshape(-1,1,order='F')
    a3 = cA3
    a3 = np.vstack((a3,a3,a3,a3,a3,a3,a3,a3))
    a3 = a3.reshape(-1,1,order='F')

    cA_1 = centered_slice(a1,len(X))
    cA_2 = centered_slice(a2,len(X))
    cA_3 = centered_slice(a3,len(X))

    cA = np.concatenate((cA_1,cA_2,cA_3),axis=1)
    cA = cA.reshape(cA.size,1,order='C')

    ind = np.argwhere(abs(cA.ravel())<np.sqrt(eps))
    cA[ind] = 0
    cA = cA.reshape(N,X.size,order='F')
    
    return cD, cA

def centered_slice(X, L):
    L = np.asarray(L)
    shape = np.array(X.shape)

#     # verify assumptions
#     assert L.shape == (X.ndim,)
#     assert ((0 <= L) & (L <= shape)).all()

    # calculate start and end indices for each axis
    starts = (shape - L) // 2
    stops = starts + L

    # convert to a single index
    idx = tuple(np.s_[a:b] for a, b in zip(starts, stops))
    return X[idx]