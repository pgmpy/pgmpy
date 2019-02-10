pgmpy
=====
[![Build Status](https://travis-ci.org/pgmpy/pgmpy.svg?style=flat)](https://travis-ci.org/pgmpy/pgmpy)
[![Appveyor](https://ci.appveyor.com/api/projects/status/github/pgmpy/pgmpy?branch=dev)](https://www.appveyor.com/)
[![codecov](https://codecov.io/gh/pgmpy/pgmpy/branch/dev/graph/badge.svg)](https://codecov.io/gh/pgmpy/pgmpy)
[![Code Health](https://landscape.io/github/pgmpy/pgmpy/dev/landscape.svg?style=flat)](https://landscape.io/github/pgmpy/pgmpy/dev)
[![Join the chat at https://gitter.im/pgmpy/pgmpy](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/pgmpy/pgmpy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

pgmpy is a python library for working with Probabilistic Graphical Models.  

Documentation  and list of algorithms supported is at our official site http://pgmpy.org/  
Examples on using pgmpy: https://github.com/pgmpy/pgmpy/tree/dev/examples  
Basic tutorial on Probabilistic Graphical models using pgmpy: https://github.com/pgmpy/pgmpy_notebook  

Our mailing list is at https://groups.google.com/forum/#!forum/pgmpy .

We have our community chat at [gitter](https://gitter.im/pgmpy/pgmpy).

Dependencies
=============
pgmpy has following non optional dependencies:
- Python 2.7 or Python 3
- NetworkX 1.11 
- Scipy 0.18.0 
- Numpy 1.11.1 
- Pandas 0.18.1 

Installation
=============
Using conda:
```
$ conda install -c ankurankan pgmpy
```

Using pip:
```
$ pip install -r requirements.txt  # or requirements-dev.txt if you want to run unittests
$ pip install pgmpy
```

Or for installing the latest codebase:
```
$ git clone https://github.com/pgmpy/pgmpy 
$ cd pgmpy/
$ pip install -r requirements.txt
$ python setup.py install
```

If you face any problems during installation let us know, via issues, mail or at our gitter channel.

Development
============

Code
----

You can check the latest sources from our github repository 
use the command:

    git clone https://github.com/pgmpy/pgmpy.git

Contributing
------------
Issues can be reported at our [issues section](https://github.com/pgmpy/pgmpy/issues).

Before opening a pull request, please have a look at our [contributing guide](
https://github.com/pgmpy/pgmpy/blob/dev/Contributing.md)

Contributing guide contains some points that will make our life's easier in reviewing and merging your PR.

If you face any problems in pull request, feel free to ask them on the mailing list or gitter.

If you want to implement any new features, please have a discussion about it on the issue tracker or the mailing
list before starting to work on it.

Testing
-------

After installation, you can launch the test form pgmpy
source directory (you will need to have the ``nose`` package installed):
```bash
$ nosetests -v
```
to see the coverage of existing code use following command
```
$ nosetests --with-coverage --cover-package=pgmpy
```

Documentation and usage
=======================

Everything is at:
http://pgmpy.org/

You can also build the documentation in your local system. We use sphinx to help us building documentation from our code.
```
$ cd /path/to/pgmpy/docs
$ make html
```
Then the docs will be in _build/html

Examples:
=========
We have a few example jupyter notebooks here: https://github.com/pgmpy/pgmpy/tree/dev/examples
For more detailed jupyter notebooks and basic tutorials on Graphical Models check: https://github.com/pgmpy/pgmpy_notebook/

License
=======
pgmpy is released under MIT License. You can read about our license at [here](https://github.com/pgmpy/pgmpy/blob/dev/LICENSE)

