from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(ext_modules=cythonize("./XXZED.pyx",language="c++"),include_dirs=[numpy.get_include()])
#setup(ext_modules = cythonize("binarybasis.pyx",language="c++"))
