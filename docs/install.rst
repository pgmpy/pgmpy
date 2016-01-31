Installation
============

Getting the dependencies
------------------------

Installing from source requires you to have installed

* :code:`Python3`
* :code:`networkx` (==1.8.1)
* :code:`numpy` (==1.9.1)
* :code:`scipy` (==0.14.0)
* :code:`cython` (==0.21)
* :code:`pandas` (==0.15.1)
* :code:`setuptools`
* working :code:`C` and :code:`C++` compiler

You can install all these requirements by issuing ::

    $ [sudo] apt-get install build-essential python3-dev python3-pip
    $ [sudo] pip3 install -r requirements.txt       # use requirements-dev.txt if you want to run tests

On Red Hat and clones (e.g CentOS), install the dependencies using::

    $ [sudo] yum -y install gcc gcc-c++ python3-devel python3-pip
    $ [sudo] pip3 install -r requirements.txt       # use requirements-dev.txt if you want to run tests

Or  use some cross-platform binary package manager such as conda_ (it is
recommended as well as the most easiest and hastle-free way)

.. _conda: http://conda.pydata.org/

Setup a virtual environment in :code:`conda` by ::

    $ conda create -n pgmpy-env python=3.4
    $ source activate pgmpy-env

Once you have the virtual environment setup, install the depenedencies using::

    $ conda install -f requirements.txt     # use requirements-dev.txt if you want to run tests

.. note::

   In order to build the documentation you will need :code:`sphinx` and to run the tests you will need :code:`nose` ::

      $ [sudo] pip3 install sphinx nose

Installing from source
----------------------

You can install from source by downloading a source archive file (zip) or by checking out the
source files from :code:`git` source repository.

1. Download the source (zip file) from https://github.com/pgmpy/pgmpy or clone the pgmpy repository::

    $ git clone https://github.com/pgmpy/pgmpy
    $ git checkout dev

2. Unpack (if necessary) and change directory to the source directory.

3. Run::

    $ [sudo] python3 setup.py install

Testing
-------

Testing requires having the :code:`nose` library. After installation, the package can be tested by executing
*from* the source directory::

    $ nosetests3

This would give you a lot of output (and some warnings) but eventually should finish without errors. Otherwise, please consider
posting an issue into the `bug tracker <https://github.com/pgmpy/pgmpy/issues>`_ or the Mailing List pgmpy@googlegroups.com .
