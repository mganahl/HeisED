#!/usr/bin/env python
import functools as fct
import numpy as np
import XXZED as ed
import scipy as sp
import time
import LanczosEngine as lanEn

comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))


"""
                      for testing only                       
"""
from scipy.sparse import csc_matrix

if __name__ == "__main__":
    """
    ED calculation of groundstate of XXZ model for a periodic 4-site chain
    """

    Jz=1.1 
    Jxy=1.0
    sx=np.zeros((2,2)).astype(complex)
    sx[0,1]=0.5
    sx[1,0]=0.5    
    sy=np.zeros((2,2)).astype(complex)
    sy[0,1]=1j*0.5
    sy[1,0]=-1j*0.5    
    sz=np.diag([-0.5,0.5]).astype(complex)
    eye=np.eye(2).astype(complex)    
    H=Jxy*(np.kron(np.kron(np.kron(sx,sx),eye),eye)+\
           np.kron(np.kron(np.kron(eye,sx),sx),eye)+\
           np.kron(np.kron(np.kron(eye,eye),sx),sx)+\
           np.kron(np.kron(np.kron(sx,eye),eye),sx)+\
           np.kron(np.kron(np.kron(sy,sy),eye),eye)+\
           np.kron(np.kron(np.kron(eye,sy),sy),eye)+\
           np.kron(np.kron(np.kron(eye,eye),sy),sy)+\
           np.kron(np.kron(np.kron(sy,eye),eye),sy))+\
           Jz*(np.kron(np.kron(np.kron(sz,sz),eye),eye)+\
           np.kron(np.kron(np.kron(eye,sz),sz),eye)+\
           np.kron(np.kron(np.kron(eye,eye),sz),sz)+\
           np.kron(np.kron(np.kron(sz,eye),eye),sz))
    eta,u=np.linalg.eigh(H)
    print(eta)
