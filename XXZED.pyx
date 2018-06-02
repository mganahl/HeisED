import numpy as np
cimport numpy as np
from scipy.special import binom
from sys import stdout

ctypedef np.uint64_t ITYPE_t
ctypedef np.float64_t DTYPE_t

"""
generate a list of uint64 representing the basis states for an N-site spin 1/2 
system, with Nup up spins and N-Nup down spins.
returns: a list of uint64. the binary representation of each number corresponds to
a spin configuration
"""
def binarybasis(N, N_up):
    cdef int dim = int(binom(N, N_up))
    cdef long unsigned int count,state
    cdef int k,nleft,p

    cdef np.ndarray[ITYPE_t, ndim=1] basis = np.empty([dim],dtype=np.uint64)
    if N_up == 0: 
        basis[0]=np.uint64(0)
        return basis

    '''
    generate the initial state by setting the first N_up bits to 1:
    '''
    state=0
    for k in range(N_up):
        state=setBit(state,k)
    count = 0
    while state!= (2**N-1)-(2**N_up-1):

        basis[count] = state
        
        '''
        make the smalles possible increase: find the first non-zero bit that can be shifted forward by one site. let n be the site of this bit, then
        shift it from n to n+1. after that, take all non-zero bits at position n'<n and move them all back to the beginning
        '''
        k=0
        nleft=0
        while True:
            if (getBit(state,k)==1):
                if (getBit(state,k+1)==1):
                    nleft+=1
                elif (getBit(state,k+1)==0):		    
                    break
            k+=1
        state=setBit(state,k+1)
        for p in range(nleft):
            state=setBit(state,p)
        for p in range(nleft,k+1):
            state=unsetBit(state,p)
        count = count + 1
    '''
    don't forget to add the last state
    '''
    basis[count]=state
    return basis


"""
generate a list of uint64 representing the basis states for an N-site spin 1/2 
system, with Nup up spins and N-Nup down spins. Uses recursive function call
returns: a list of uint64. the binary representation of each number corresponds to
a spin configuration
np.uint64(n+2**p))
"""
def binarybasisrecursive(int N,int Nup):
    cdef int p
    cdef list basis=[]
    cdef list init=[np.uint64(0)]
    cdef list rest
    cdef long unsigned int n
    if Nup==0:
        return init
    else:
        for p in range(Nup-1,N):
            rest=binarybasisrecursive(p,Nup-1)
            for n in rest:
                basis.append(n+setBit(0,p))
        return basis



"""
flips a bit at position pos; b has to be an unsigned integer of 64 bit! (i.e. np.uint64)
"""
cdef long unsigned int flipBit(unsigned long int b,unsigned int pos):
    cdef long unsigned int mask
    cdef long unsigned int a=1
    mask=a<<pos
    if b&mask==0:#b(pos)==0
        return b|mask
    elif b&mask!=0:#b(pos)==1
        return b&(~mask)

"""
sets a bit at position pos; b has to be an unsigned integer of 64 bit! (i.e. np.uint64)
"""
cdef long unsigned int setBit(long unsigned int b, unsigned int  pos):
    cdef long unsigned int mask
    cdef long unsigned int a=1
    mask=a<<pos
    return mask|b


"""
sets a bit at position pos; b has to be an unsigned integer of 64 bit! (i.e. np.uint64)
"""
cdef long unsigned int unsetBit(long unsigned int b, unsigned int  pos):
    cdef long unsigned int mask
    cdef long unsigned int a=1
    mask=a<<pos
    return (~mask)&b

cdef int getBit(unsigned long int  b,unsigned int pos):
    cdef long unsigned int mask
    cdef long unsigned int a=1
    mask = a << pos
    return int((b & mask)>0)


"""
calculates all non-zero matrix elements of the XXZ Hamiltonian on a grid "grid" with interactions
Jz and Jxy and for total number of N spins. "basis" is a list of unit64 numbers encoding the basis-states

grid (list() of list()) of length N: grid[n] is a list of neighbors of spin n

Jz,Jxy: Jz[n] and Jxy[n] is an array of the interaction and hopping parameters of all neighbors of spin n,
such that Jz[n][i] corresponds to the interaction of spin n with spin grid[n][i] (similar for Jxy)

RETURNS: inddiag,diag,nondiagindx,nondiagindy,nondiag

diag:   a list of non-zero diagonal element from the Jz Sz*Sz -part of the Hamiltonian
inddiag: a list indices of the non-zero matrix elements from the Sz*Sz part, such that H[inddiag[n],inddiag[n]]=diag[n]


nondiag:   a list of non-zero matrix elements form the Sx*Sx+Sy*Sy-part of the Hamiltonian
nondiagindx,nondiagindy: a list x- and y-indices of the non-zero values form the Jxy(Sx*Sx+Sy*Sy) part
                         such that H[nondiagindx[n],nondiagindy[n]]=nondiag[n]

"""
def XXZGrid(np.ndarray[DTYPE_t, ndim=2] Jxy,np.ndarray[DTYPE_t, ndim=2] Jz,int N,basis,grid):
    num2ind={}
    cdef long unsigned int state,newstate
    cdef int n,N0,s,p,nei
    cdef float sz,szsz
    for n in range(len(basis)):
        num2ind[basis[n]]=n
    cdef np.ndarray[DTYPE_t, ndim=2] Jp=Jxy/2.0
    cdef list diag=[]
    cdef list inddiag=[]
    cdef list nondiag=[]
    cdef list nondiagindx=[]
    cdef list nondiagindy=[]
    N0=len(basis)
    for n in range(len(basis)):
        if n%10000==0:
            stdout.write("\r building Hamiltonian ... finished %2.2f percent" %(100.0*n/N0))
            stdout.flush()
        state=basis[n]
        szsz=0
        for s in range(N):
            sz=(getBit(state,s)-0.5)*2
            for p in range(len(grid[s])):
                nei=grid[s][p]
                szsz+=sz*(getBit(state,nei)-0.5)*2*Jz[s,p]/4.0
        if (abs(szsz)>1E-5):
            diag.append(szsz)
            inddiag.append(n)
        for s in range(N):
            for p in range(len(grid[s])):
                nei=grid[s][p]
                if getBit(state,s)!=getBit(state,nei):
                    newstate=flipBit(flipBit(state,s),nei)
                    nondiagindx.append(num2ind[newstate])
                    nondiagindy.append(n)
                    nondiag.append(Jp[s,p])
    stdout.write("\r building Hamiltonian ... finished %2.2f percent" %(100.0*n/N0))
    stdout.flush()
    print()
    return inddiag,diag,nondiagindx,nondiagindy,nondiag

def testbinops(unsigned long int b,int pos):
    print(bin(~b))
    print('b={2}, bit {0} of b={1}'.format(pos,getBit(b,pos),bin(b)))
    print('b before flipping bit {0}:'.format(pos),bin(b))
    print('b after flipping bit {0}:'.format(pos),bin(flipBit(b,pos)))
    print('b before setting bit {0}:'.format(pos),bin(b))
    print('b after setting bit {0}:'.format(pos),bin(setBit(b,pos)))
