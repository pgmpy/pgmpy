pgmpy
=====
[![Build Status](https://travis-ci.org/pgmpy/pgmpy.svg?style=flat)](https://travis-ci.org/pgmpy/pgmpy)
[![Appveyor](https://ci.appveyor.com/api/projects/status/github/pgmpy/pgmpy?branch=dev)](https://www.appveyor.com/)
[![Coverage Status](https://coveralls.io/repos/pgmpy/pgmpy/badge.svg?branch=dev)](https://coveralls.io/r/pgmpy/pgmpy?branch=dev)
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

Download
=========
Currently pgmpy is not hosted on pypi or conda.
You can either clone the git repo with:
```
git clone https://github.com/pgmpy/pgmpy
```
or download a zip from: https://github.com/pgmpy/pgmpy/archive/dev.zip

Installation
=============
To install the dependencies switch to the pgmpy directory using:
```
$ cd /path/to/pgmpy
```
In the directory run either of the following:

Using pip
```
$ pip install -r requirements.txt  # or requirements-dev.txt if you want to run unittests
```
or conda
```
$ conda install --file requirements.txt  # or requirements-dev.txt
```

Then install using:

```bash
sudo python setup.py install
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
Issues can be reported at our [issues section](https://github.com/pgmpy/pgmpy/issues) or via mail, or gitter.
We will try our best to solve the issue at the earliest.

Before opening a pull request , have look at our [contributing guide](
https://github.com/pgmpy/pgmpy/blob/dev/Contributing.md)

Contributing guide contains some points that will make our life's easier in reviewing and merging your PR.

If you face any problems in pull request, feel free to ask them at mail or gitter.

If you have any new features, please have a discussion on the issue tracker or the mailing
list before working on it.

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

Example:
========
Here is a small snippet of pgmpy API
```python
>>> from pgmpy.models import BayesianModel
>>> from pgmpy.factors.discrete import TabularCPD
>>> student = BayesianModel()
>>> # instantiates a new Bayesian Model called 'student'

>>> student.add_nodes_from(['diff', 'intel', 'grade'])
>>> # adds nodes labelled 'diff', 'intel', 'grade' to student

>>> student.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
>>> # adds directed edges from 'diff' to 'grade' and 'intel' to 'grade'

>>> """
... diff cpd:
...
... +-------+--------+
... |diff:  |        |
... +-------+--------+
... |easy	|	0.2	 |
... +-------+--------+
... |hard	|	0.8	 |
... +-------+--------+
... """

>>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])

>>> """
... intel cpd:
...
... +-------+--------+
... |intel: |        |
... +-------+--------+
... |dumb	|	0.5	 |
... +-------+--------+
... |avg	|	0.3	 |
... +-------+--------+
... |smart	|	0.2	 |
... +-------+--------+
... """

>>> intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])

>>> """
... grade cpd:
...
... +------+-----------------------+---------------------+
... |diff: |          easy         |         hard        |
... +------+------+------+---------+------+------+-------+
... |intel:| dumb |  avg |  smart  | dumb | avg  | smart |
... +------+------+------+---------+------+------+-------+
... |gradeA| 0.1  | 0.1  |   0.1   |  0.1 |  0.1 |   0.1 |
... +------+------+------+---------+------+------+-------+
... |gradeB| 0.1  | 0.1  |   0.1   |  0.1 |  0.1 |   0.1 |
... +------+------+------+---------+------+------+-------+
... |gradeC| 0.8  | 0.8  |   0.8   |  0.8 |  0.8 |   0.8 |
... +------+------+------+---------+------+------+-------+
... """

>>> grade_cpd = TabularCPD('grade', 3,
					     [[0.1,0.1,0.1,0.1,0.1,0.1],
                         [0.1,0.1,0.1,0.1,0.1,0.1], 
                         [0.8,0.8,0.8,0.8,0.8,0.8]],
					     evidence=['intel', 'diff'],
					     evidence_card=[3, 2])

>>> student.add_cpds(diff_cpd, intel_cpd, grade_cpd)

>>> # Finding active trail
>>> student.active_trail_nodes('diff')
{'diff', 'grade'}

>>> # Finding active trail with observation
>>> student.active_trail_nodes('diff', observed='grade')
{'diff', 'intel'}

```
License
=======
pgmpy is released under MIT License. You can read about our lisence at [here](https://github.com/pgmpy/pgmpy/blob/dev/LICENSE)

