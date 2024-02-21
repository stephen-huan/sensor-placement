from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html
libraries = [
    "mkl_intel_lp64",
    "mkl_intel_thread",
    "mkl_core",
    "iomp5",
    "pthread",
    "m",
    "dl"
]

extensions = [
    Extension(
        "cysensor/*", ["cysensor/*.pyx"],
        # to cimport numpy
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        libraries=libraries,
        # extra_compile_args=["-Ofast", "-ffast-math"],
    ),
]

setup(
    ext_modules=\
    cythonize(extensions,
              annotate=True,
              compiler_directives={
                  "language_level": 3,
                  "boundscheck": False,
                  "wraparound": False,
                  "initializedcheck": False,
                  "cdivision": True,
              },
    ),
)
