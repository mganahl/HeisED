# HeisED
Python code for exact diagonalization of the Heisenberg XXZ model on 1d and 2d grids.
The Hamiltonian is built and stored in scipy.sparse.csc_matrix format. Generation of
all matrix elements is done in cython.
The code uses total Sz and Z2 conservation to reduce Hilbertspace dimension. main.py 
currently runs diagonalization on a 1d XXZ chain for either open or periodic boundary conditions;
run python main.py --h for a list of parameters


To build .so files, run python setup.py built_ext --inplace


