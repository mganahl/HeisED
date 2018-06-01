# HeisED
Python code for exact diagonalization of the Heisenberg XXZ model on 1d and 2d grids.
The Hamiltonian is built and stored in scipy.sparse.csc_matrix format. Generation of
all matrix elements is done in cython.

To build .so files, run python setup.py built_ext --inplace


