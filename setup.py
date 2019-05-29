#from distutils.core import setup, Extension
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

ext = Extension("spherical_util",
        sources=["spherical_util.pyx","spherical_utils.cpp"],
        language="c++",
        include_dirs=[np.get_include()])

setup(name="density_builder",
        ext_modules=cythonize(ext))
