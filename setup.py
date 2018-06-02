from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize("./XXZED.pyx",language="c++"))
#setup(ext_modules = cythonize("binarybasis.pyx",language="c++"))
