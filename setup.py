#/usr/bin/env python3

USE_CYTHON = True

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

try:
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False

ext_modules = cythonize('pgmpy/Factor/_factor_product.pyx')
cmdclass = {'build_ext': build_ext}

if USE_CYTHON:
    ext_modules.extend([
        Extension("pgmpy.Factor._factor_product",
                  ["pgmpy/Factor/_factor_product.pyx"])
    ])
else:
    ext_modules.append(
        Extension("pgmpy.Factor._factor_product",
                  ["pgmpy/Factor/_factor_product.c"])
    )

setup(
    name="pgmpy",
    packages=["pgmpy",
              "pgmpy.BayesianModel",
              "pgmpy.Exceptions",
              "pgmpy.Factor",
              "pgmpy.Independencies",
              "pgmpy.MarkovModel",
#              "pgmpy.readwrite"
            ],
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
#    install_requires=[
#        "networkx >= 1.8.1",
#        "scipy >= 0.12.1",
#        "numpy >= 1.7.1"
#    ]
)
