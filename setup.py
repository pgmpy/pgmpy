#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="pgmpy",
    version="0.1.0",
    description="A library for Probabilistic Graphical Models",
    packages=find_packages(exclude=['tests']),
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
        "numpy >= 1.7.0",
        "nose >= 1.3.0",
        "coveralls >= 0.4"
    ],
)
