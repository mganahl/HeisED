from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize("./XXZED.pyx"))
#setup(ext_modules = cythonize("binarybasis.pyx",language="c++"))
