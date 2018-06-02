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

from scipy.sparse import csc_matrix

if __name__ == "__main__":
    """
    ED calculation of groundstate of XXZ model. You can use either arnoldi (scipy implementation) 
    or a lanczos (my implementation). The latter is faster for large matrices
    """
    N=30 #system size
    Nup=5 #number of up-spins
    Jz=1.0 
    Jxy=1.0
    LAN=False
    save=False
    filename='XXZsparseN{0}Nup{1}Jz{2}Jxy{3}'.format(N,Nup,Jz,Jxy)
    print('running ED for N={0}, Nup={1}, Jz={2}, Jxy={3}'.format(N,Nup,Jz,Jxy))
    #create the basis states, encoded as list of long unsigned int
    t1=time.time()
    basis=ed.binarybasis(N, Nup)
    t2=time.time()
    print('calculating basis-states took {0} seconds'.format(t2-t1))
    D=len(basis)
    print('basis size: {0}'.format(D))

    #grid: a list of length N; grid[n] is a list of neighbors of spin n
    grid=[None]*N

    #Jzar,Jxyar: a list of length N; Jz[n] and Jxy[n] is an array of the interaction and hopping parameters of all neighbors of spin n,
    #such that Jz[n][i] corresponds to the interaction of spin n with spin grid[n][i]

    Jzar=[0.0]*N
    Jxyar=[0.0]*N
    for n in range(N-1):
        grid[n]=[n+1]
        Jzar[n]=np.asarray([Jz])
        Jxyar[n]=np.asarray([Jxy])
    grid[N-1]=[]
    Jzar[N-1]=np.asarray([0.0])
    Jxyar[N-1]=np.asarray([0.0])
    Jxyar,Jzar=np.asarray(Jxyar).astype(np.float64),np.asarray(Jzar).astype(np.float64)
    t3=time.time()
    #inddiag,diag,nondiagindx,nondiagindy,nondiag=ed.XXZGrid2(Jxyar,Jzar,N,basis,grid)
    inddiag,diag,nondiagindx,nondiagindy,nondiag=ed.XXZGrid(Jxyar,Jzar,N,basis,grid)


    Hsparse=csc_matrix((diag,(inddiag, inddiag)),shape=(len(basis), len(basis)))+csc_matrix((nondiag,(nondiagindx, nondiagindy)),shape=(len(basis), len(basis)))
    t4=time.time()
    print('calculating sparse Hamiltonian took {0} seconds'.format(t4-t3))
    if save:
        sp.sparse.save_npz(filename,Hsparse)
    t5=time.time()        
    if LAN==False:
        e,v=sp.sparse.linalg.eigsh(Hsparse,k=10,which='SA',maxiter=1000000,tol=1E-5,v0=None,ncv=40)
        print('lowest energies:')
        print(e)

    if LAN==True:
        def matvec(mat,vec):
            return mat.dot(vec)
        mv=fct.partial(matvec,*[Hsparse])
        lan=lanEn.LanczosEngine(mv,np.dot,np.zeros,Ndiag=10,ncv=500,numeig=10,delta=1E-8,deltaEta=1E-10)
        el,vl,conv=lan.__simulate__(np.random.rand(D),verbose=False)
        print('lowest energies:')
        print(el)
    t6=time.time()
    print('sparse diagonalizatoin took {0} seconds'.format(t6-t5))
