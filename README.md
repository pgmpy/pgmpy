pgmpy
=====
![Build](https://github.com/pgmpy/pgmpy/actions/workflows/ci.yml/badge.svg?branch=dev)
[![codecov](https://codecov.io/gh/pgmpy/pgmpy/branch/dev/graph/badge.svg)](https://codecov.io/gh/pgmpy/pgmpy)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/78a8256c90654c6892627f6d8bbcea14)](https://www.codacy.com/gh/pgmpy/pgmpy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pgmpy/pgmpy&amp;utm_campaign=Badge_Grade)
[![Downloads](https://img.shields.io/pypi/dm/pgmpy.svg)](https://pypistats.org/packages/pgmpy)
[![Join the chat at https://gitter.im/pgmpy/pgmpy](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/pgmpy/pgmpy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](http://pgmpy.org/pgmpy-benchmarks/)

pgmpy is a Python package for working with Bayesian Networks and related models such as Directed Acyclic Graphs, Dynamic Bayesian Networks, and Structural Equation Models. It combines features from both causal inference and probabilistic inference literatures to allow users to seamlessly work between both. It implements algorithms for structure learning/causal discovery, parameter estimation, probabilistic and causal inference, and simulations.

The documentation is available at: https://pgmpy.org/
Installation instructions are available at: https://pgmpy.org/started/install.html

Example notebooks are available at: https://github.com/pgmpy/pgmpy/tree/dev/examples

Tutorials on Bayesian Networks: https://github.com/pgmpy/pgmpy_notebook

Our mailing list is at https://groups.google.com/forum/#!forum/pgmpy .

We have our community chat at [gitter](https://gitter.im/pgmpy/pgmpy).

Installation
=============
To install pgmpy from the source code:
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
Our latest codebase is available on the `dev` branch of the repository.

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
source directory (you will need to have the ``pytest`` package installed):
```bash
$ pytest -v
```
to see the coverage of existing code use following command
```
$ pytest --cov-report html --cov=pgmpy
```

Documentation
=============

We use sphinx to build the documentation. Please refer: https://pgmpy.org/exact_infer/bp.html for steps to build docs locally.

Examples
========
We have a few example jupyter notebooks here: https://github.com/pgmpy/pgmpy/tree/dev/examples
For more detailed jupyter notebooks and basic tutorials on Graphical Models check: https://github.com/pgmpy/pgmpy_notebook/

Citing
======
If you use `pgmpy` in your scientific work, please consider citing us:

```
Ankan, Ankur, Abinash, Panda. "pgmpy: Probabilistic Graphical Models using Python." Proceedings of the Python in Science Conference. SciPy, 2015.
```

Bibtex:
```
@inproceedings{Ankan2015,
  series = {SciPy},
  title = {pgmpy: Probabilistic Graphical Models using Python},
  ISSN = {2575-9752},
  url = {http://dx.doi.org/10.25080/Majora-7b98e3ed-001},
  DOI = {10.25080/majora-7b98e3ed-001},
  booktitle = {Proceedings of the Python in Science Conference},
  publisher = {SciPy},
  author = {Ankan,  Ankur and Panda,  Abinash},
  year = {2015},
  collection = {SciPy}
}
```

License
=======
pgmpy is released under MIT License. You can read about our license at [here](https://github.com/pgmpy/pgmpy/blob/dev/LICENSE)
