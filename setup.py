#/usr/bin/env python3
import ez_setup
ez_setup.use_setuptools()

USE_CYTHON = True

from setuptools import setup, find_packages
from distutils.extension import Extension
import numpy as np

try:
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False

ext_modules = []
cmdclass = {}

if USE_CYTHON:
    ext_modules.extend([
        Extension("pgmpy.Factor._factor_product",
                  ["pgmpy/Factor/_factor_product.pyx"])
    ])
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules.extend([
        Extension("pgmpy.Factor._factor_product",
                  ["pgmpy/Factor/_factor_product.c"])
    ])

setup(
    name="pgmpy",
    packages=find_packages(exclude=['tests']),
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    version="0.1.0",
    author=open("AUTHORS.rst").read(),
    url="https://github.com/pgmpy/pgmpy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Researchers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering"
    ],
    long_description=open("README.md").read(),
    install_requires=[
        "networkx >= 1.8.1",
        "scipy >= 0.12.1",
#        "numpy >= 1.7.1",
        "nose >= 1.3.0",
        "coveralls >= 0.4"
    ],
    include_dirs = [np.get_include()]
)
