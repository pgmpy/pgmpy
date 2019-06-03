#!/usr/bin/env python3

from setuptools import setup, find_packages

import pgmpy

setup(
    name="pgmpy",
    version=pgmpy.__version__,
    description="A library for Probabilistic Graphical Models",
    packages=find_packages(exclude=["tests"]),
    author="Ankur Ankan",
    author_email="ankurankan@gmail.com",
    url="https://github.com/pgmpy/pgmpy",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering",
    ],
    long_description="https://github.com/pgmpy/pgmpy/blob/dev/README.md",
    install_requires=[],
)
