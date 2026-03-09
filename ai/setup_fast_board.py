"""Build script for C-accelerated board reconstruction.

Usage:
    python ai/setup_fast_board.py build_ext --inplace
"""
from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    extensions = cythonize([
        Extension(
            "ai._fast_board",
            sources=["ai/_fast_board.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    ], compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'language_level': '3',
    })
except ImportError:
    print("Cython not found. Skipping C extension build.")
    extensions = []

setup(
    name="ai_fast_board",
    ext_modules=extensions,
)
