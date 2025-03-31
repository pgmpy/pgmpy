.. pgmpy documentation master file, created by
   sphinx-quickstart on Tue Aug 30 18:17:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />

.. image:: https://github.com/pgmpy/pgmpy/actions/workflows/ci.yml/badge.svg?branch=dev
   :target: https://github.com/pgmpy/pgmpy/actions?query=branch%3Adev

.. image:: https://img.shields.io/pypi/dm/pgmpy.svg
   :target: https://pypistats.org/packages/pgmpy

.. image:: https://img.shields.io/pypi/v/pgmpy?color=blue
   :target: https://pypi.org/project/pgmpy/

.. image:: https://img.shields.io/pypi/pyversions/pgmpy.svg?color=blue
   :target: https://pypi.org/project/pgmpy/

.. image:: https://img.shields.io/github/license/pgmpy/pgmpy
   :target: https://github.com/pgmpy/pgmpy/blob/dev/LICENSE

.. image:: http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat
   :target: http://pgmpy.org/pgmpy-benchmarks/

.. |br| raw:: html

   <br />
   <br />


.. image:: https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white
   :align: center
   :target: https://discord.gg/DRkdKaumBs

.. toctree::
   :maxdepth: 2
   :hidden:

   started/base.rst
   base/base.rst
   models/base.rst
   factors/base.rst
   exact_infer/base.rst
   approx_infer/base.rst
   param_estimator/base.rst
   structure_estimator/base.rst
   metrics/metrics.rst
   readwrite/base.rst
   plotting.rst
   examples.rst
   tutorial.rst

pgmpy is a Python package for causal inference and probabilistic inference
using Directed Acyclic Graphs (DAGs) and Bayesian Networks with a focus on
modularity and extensibility. Implementations of various algorithms for Causal
Discovery (a.k.a, Structure Learning), Parameter Estimation, Approximate
(Sampling Based) and Exact inference, and Causal Inference are available.

|

.. figure:: pgmpy_workflow.png

   Possible Workflows in pgmpy for Directed Acyclic Graphs (DAGs) and Bayesian Networks (BNs).

|

Supported Data Types
====================

.. list-table::
   :header-rows: 1

   * -
     - Causal Discovery
     - Parameter Estimation
     - Causal Inference
     - Probabilistic Inference
     - Simulations
   * - **Categorical**
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - **Continuous**
     - Yes
     - Yes
     - Yes (partial)
     - Yes
     - Yes
   * - **Mixed**
     - Yes
     - No
     - No
     - No
     - Yes
   * - **Time Series**
     - No
     - Yes
     - Yes (ApproximateInference)
     - Yes
     - Yes

|

Algorithms
==========

.. csv-table::
   :file: algorithms.csv
   :header-rows: 1

|

Examples
========

**Example notebooks:** https://pgmpy.org/examples.html

**Tutorial notebooks:** https://pgmpy.org/tutorial.html

|

Citation
========
If you use pgmpy in your scientific work, please consider citing us:

.. code-block:: text

   Ankur Ankan, & Johannes Textor (2024). pgmpy: A Python Toolkit for Bayesian Networks. Journal of Machine Learning Research, 25(265), 1–8.

Bibtex:

.. code-block:: text

   @article{Ankan2024,
     author  = {Ankur Ankan and Johannes Textor},
     title   = {pgmpy: A Python Toolkit for Bayesian Networks},
     journal = {Journal of Machine Learning Research},
     year    = {2024},
     volume  = {25},
     number  = {265},
     pages   = {1--8},
     url     = {http://jmlr.org/papers/v25/23-0487.html}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
