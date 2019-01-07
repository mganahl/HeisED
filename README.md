# HeisED
dev branch: up next: translational invariance implementation

Python/Cython code for exact diagonalization of the Heisenberg XXZ model on 1d and 2d grids.
The Hamiltonian is built and stored in scipy.sparse.csc_matrix format. Generation of
all matrix elements is done in cython.
The code uses total Sz and Z2 conservation to reduce Hilbertspace dimension. main.py 
currently runs diagonalization on a 1d XXZ chain for either open or periodic boundary conditions;
run python main.py --h for a list of parameters

To build .so files, run ```python setup.py built_ext --inplace```

Tested for python2.7.12 (numpy 1.12.1,scipy 0.19.0) and python3.5.2 (numpy 1.14.3, scipy 0.19.0)
