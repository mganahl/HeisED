#!/usr/bin/env python
import numpy as np
import copy




"""
This is a general purpose Lanczos-class. It performs a Lanczos tridiagonalization 
of a Hamiltonian, defined by the matrix-vector product matvec. matvec has to take
an object "vector" and return matvec(vector). vector can be any ndarray (vector or tensor).
Ndiag is the iteration step at which diagonalization of the tridiagonal Hamiltonian 
should be done, e.g. Ndiag=4 means every 4 steps the tridiagonal Hamiltonian is diagonalized
and it is checked if the eigenvalues are sufficiently converged.
nmax is the maximum number of steps
numeig is the number of eigenvalue-eigenvector pairs to be returned by the routine
delta is a tolerance parameter, such that iteration stops when a vector with norm<delta
is found.
deltaEta is the desired eigenvalue-accuracy.
"""
class LanczosEngine:
    def __init__(self,matvec,vecvec,zeros_initializer,Ndiag,ncv,numeig,delta,deltaEta):
        assert(ncv>=numeig)
        self._Ndiag=Ndiag
        self._ncv=ncv
        self._numeig=numeig
        self._delta=delta
        self._deltaEta=deltaEta
        self._matvec=matvec
        self._dot=vecvec
        self._zeros=zeros_initializer
        assert(Ndiag>0)
        
        
    def __simulate__(self,initialstate,verbose=False):
        Dim=1
        for d in initialstate.shape:
            Dim*=d
        #initialization:
        xn=copy.deepcopy(initialstate)
        xn/=np.sqrt(self._dot(xn.conjugate(),xn))
        xn_minus_1=self._zeros(initialstate.shape,dtype=initialstate.dtype)
        converged=False
        it=0
        kn=[]
        epsn=[]
        self._vecs=[]
        first=True
        while converged==False:
            #normalize the current vector:
            knval=np.sqrt(self._dot(xn.conjugate(),xn))

            if knval<self._delta:
                converged=True
                break
            kn.append(knval)
            xn=xn/kn[-1]
            #store the Lanczos vector for later
            self._vecs.append(xn)
            Hxn=self._matvec(xn)
            epsn.append(self._dot(xn.conjugate(),Hxn))
            if ((it%self._Ndiag)==0)&(len(epsn)>=self._numeig):
                #diagonalize the effective Hamiltonian
                Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
                eta,u=np.linalg.eigh(Heff)
                if first==False:
                    if np.linalg.norm(eta[0:self._numeig]-etaold[0:self._numeig])<self._deltaEta:
                        converged=True
                first=False
                etaold=eta[0:self._numeig]

            if it>0:
                Hxn-=(self._vecs[-1]*epsn[-1])
                Hxn-=(self._vecs[-2]*kn[-1])
            else:
                Hxn-=(self._vecs[-1]*epsn[-1])
            xn=Hxn
            it=it+1
            if it>self._ncv:
                break

        #now get the eigenvectors with the lowest eigenvalues:
        self._Heff=np.diag(epsn)+np.diag(kn[1:],1)+np.diag(np.conj(kn[1:]),-1)
        eta,u=np.linalg.eigh(self._Heff)
        states=[]
        for n2 in range(min(self._numeig,len(eta))):
            state=np.zeros(initialstate.shape,dtype=initialstate.dtype)
            for n1 in range(len(self._vecs)):
                state+=self._vecs[n1]*u[n1,n2]
            states.append(state/np.sqrt(self._dot(state.conjugate(),state)))
        return eta[0:min(self._numeig,len(eta))],states
                