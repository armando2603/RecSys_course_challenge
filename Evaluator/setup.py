"""
Created on 05/12/19

@author: Giuseppe Serna
"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("evaluation.pyx")
)
