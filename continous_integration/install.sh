#!/usr/bin/env bash
# This script is inspired from **scikit-learn** implementation of continous test
# integration. This is meant to "install" all the packages required for installing
# pgmpy.

# License: The MIT License (MIT)

set -e

sudo apt-get update -qq
sudo apt-get install build-essential -qq

if [[ "$DISTRIB" == "conda" ]]; then
	# Deactivate the travis-provided virtual environment and setup a
	# conda-based environment instead
	deactivate

	# Use the miniconda installer for faster download / install of conda
	# itself
    wget http://repo.continuum.io/miniconda/Miniconda-2.2.2-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/anaconda/bin:$PATH
    conda update --yes conda

	conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
		numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION networkx=$NETWORKX_VERSION \
		pandas cython
    source activate testenv
fi

if [[ "$COVERAGE" == "true" ]]; then
	pip install coverage coveralls
fi

# Build pgmpy
python setup.py develop
