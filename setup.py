#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="pgmpy",
    version="0.1.7",
    description="A library for Probabilistic Graphical Models",
    packages=find_packages(exclude=['tests']),
    author="Ankur Ankan",
    author_email="ankurankan@gmail.com",
    url="https://github.com/pgmpy/pgmpy",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 2.7",
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering"
    ],
    long_description="https://github.com/pgmpy/pgmpy/blob/dev/README.md",
    install_requires=[
        "networkx >= 1.11, <1.12",
        "scipy >= 1.0.0",
        "numpy >= 1.14.0",
    ],
)
