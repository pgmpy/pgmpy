GSoC 2014 Ideas
---------------

Introduction
~~~~~~~~~~~~

Probabilistic Graphical Models (PGM) use graphs to denote the conditional dependence structure between random variables.
They are most commonly used in probability theory, statistics (particularly Bayesian statistics) and machine learning.

pgmpy is a Python library to implement Probabilistic Graphical Models and related algorithms.
The main focus is on providing a consistent API and flexible approach to its implementation.
This is the first time pgmpy is applying for GSoC under the Python Software Foundation's umbrella.

Want to get involved?
~~~~~~~~~~~~~~~~~~~~~

If you're interested in participating in GSoC 2014 as a student, mentor, or interested community member, you should join the pgmpy's mailing
list and post any questions, comments, etc. to pgmpy@googlegroups.com

You can also contact the mentors with your ideas.

Anavil Tripathi: anaviltripathi@gmail.com

Shikhar Nigam: snigam3112@gmail.com

Soumya Kundu: samkent.1729@gmail.com

Additionally, you can find us on IRC at #pgmpy on irc.freenode.org.
If no one is available to answer your question, please be patient and post it to the mailing list as well.

Getting Started
~~~~~~~~~~~~~~~

Reference book for PGM: `Probabilistic Graphical Models - Principles and Techniques <http://www.amazon.in/Probabilistic-Graphical-Models-Principles-Computation/dp/0262013193>`_

pgmpy
*****

1. Install dependencies::

    $ sudo pip3 install networkx numpy scipy cython

2. Clone the repo::

    $ git clone https://github.com/pgmpy/pgmpy

3. Install pgmpy::

    $ cd pgmpy/
    $ sudo python3 setup.py install

pgmpy_viz
*********

1. Install dependencies::

    $ sudo pip3 install django

2. Clone the repo::

    $ git clone https://github.com/pgmpy/pgmpy_viz

3. Run local server::


    $ cd pgmpy_viz/
    $ python3 manage.py runserver

Go to :code:`localhost:8000` in your browser to access the pgmpy_viz page.

Example
~~~~~~~
::

    from pgmpy.models import BayesianModel
    from pgmpy.factors import TabularCPD
    student = bm.BayesianModel()
    # instantiates a new Bayesian Model called 'student'

    student.add_nodes_from(['diff', 'intel', 'grade'])
    # adds nodes labelled 'diff', 'intel', 'grade' to student

    student.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
    # adds directed edges from 'diff' to 'grade' and 'intel' to 'grade'

    """
    diff cpd:

    +-------+--------+
    |diff:  |        |
    +-------+--------+
    |easy   |   0.2  |
    +-------+--------+
    |hard   |   0.8  |
    +-------+--------+
    """
    diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])

    """
    intel cpd:

    +-------+--------+
    |intel: |        |
    +-------+--------+
    |dumb   |   0.5  |
    +-------+--------+
    |avg    |   0.3  |
    +-------+--------+
    |smart  |   0.2  |
    +-------+--------+
    """
    intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])

    """
    grade cpd:

    +------+-----------------------+---------------------+
    |diff: |          easy         |         hard        |
    +------+------+------+---------+------+------+-------+
    |intel:| dumb |  avg |  smart  | dumb | avg  | smart |
    +------+------+------+---------+------+------+-------+
    |gradeA| 0.1  | 0.1  |   0.1   |  0.1 |  0.1 |   0.1 |
    +------+------+------+---------+------+------+-------+
    |gradeB| 0.1  | 0.1  |   0.1   |  0.1 |  0.1 |   0.1 |
    +------+------+------+---------+------+------+-------+
    |gradeC| 0.8  | 0.8  |   0.8   |  0.8 |  0.8 |   0.8 |
    +------+------+------+---------+------+------+-------+
    """
    grade_cpd = TabularCPD('grade', 3,
                        [[0.1,0.1,0.1,0.1,0.1,0.1],
                            [0.1,0.1,0.1,0.1,0.1,0.1], 
                            [0.8,0.8,0.8,0.8,0.8,0.8]],
                        evidence=['diff', 'intel'],
                        evidence_card=[2, 3])

    student.add_cpds(diff_cpd, intel_cpd, grade_cpd)

    # Finding active trail
    student.active_trail_nodes('diff')

    # Finding active trail with observation
    student.active_trail_nodes('diff', observed='grades')

Ideas
~~~~~

**1. Parsing from and writing to standard PGM file formats**
************************************************************

There are various standard file formats for representing the PGM data.
PGM data basically consists of a Graph, a table corresponding to each node and a few other attributes of the Graph.
`Here <https://github.com/pgmpy/pgmpy/issues/65>`_ is a list of some of these formats. pgmpy needs functionality to read networks from and write networks to these standard file formats.
Currently only ProbModelXML is supported. pgmpy uses lxml for XML formats and we plan to use `pyparsing <http://pyparsing.wikispaces.com/>`_ for non XML formats.

**Expected Outcome**: You are expected to choose at least one file format from the above list and write a sub-module which enables pgmpy to read from and write to the same format.

**Difficulty level**: Medium

**PGM knowledge required**: Basic knowledge about representation of PGM models.

**Skills required**: Intermediate python

**Potential Mentor(s)**: Shikhar Nigam

**2. Adding features to pgmpy_viz**
***********************************

pgmpy_viz is a web application for creating and visualizing graphical models that runs pgmpy in the back-end.
It uses cytoscape.js in the front-end for manipulation of the networks. For reference to a similar application you can look at SamIam.

This project needs you to add:

* Network validation before posting data to the server.
* Options for inference from networks.
* Porting pgmpy_viz from Django to Flask.

**Expected Outcome**: You are expected to design a Flask based web application which would enable the user to visualize the outcomes of analysis of the network.

**Difficulty level**: Medium

**PGM knowledge required**: None

**Skills required**: HTML5, CSS, JavaScript, Flask

**Potential Mentor(s)**: Soumya Kundu

**3. Implementing Markov Networks**
***********************************

There are two common branches of graphical representation of distributions.
They are Bayesian networks(Directed Acyclic Graphs) and Markov networks(Undirected graphs which may be cyclic).
Currently, pgmpy supports Bayesian Networks.
The following features for Markov Networks need to be implemented:

* Create and edit Markov Networks.
* Finding reduced Markov Networks.
* Finding independencies in Markov Networks.

**Expected Outcome**: You are expected to write a sub-module implementing the above listed features.

**Difficulty level**: Hard

**PGM knowledge required**: Good understanding of Markov Networks

**Skills required**: Intermediate python, Cython

**Potential Mentor(s)**: Anavil Tripathi

**4. Implementing Algorithms:**
*******************************

PGM involves many theorems and algorithms such as Belief-Propagation, Variable Elimination etc.
The library will eventually implement every PGM algorithm. Here is the proposed set of algorithms to be implemented.

**Expected Outcome**: You are expected to select at least one algorithm from the list and implement it.

**Difficulty level**: Hard

**PGM knowledge required**: Good understanding of PGM

**Skills required**: Intermediate python, Cython

**Potential Mentor(s)**: Shikhar Nigam

**5. Blue Sky Project**
***********************

If you have any interesting ideas please discuss it over the mailing list.

Interested Students
~~~~~~~~~~~~~~~~~~~

If you are interested in participating in GSoC with pgmpy, please introduce yourself on the mailing list.
