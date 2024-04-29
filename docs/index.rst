.. pgmpy documentation master file, created by
   sphinx-quickstart on Tue Aug 30 18:17:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />

.. image:: https://github.com/pgmpy/pgmpy/actions/workflows/ci.yml/badge.svg?branch=dev
   :target: https://github.com/pgmpy/pgmpy/actions?query=branch%3Adev

.. image:: https://codecov.io/gh/pgmpy/pgmpy/branch/dev/graph/badge.svg
   :target: https://codecov.io/gh/pgmpy/pgmpy

.. image:: https://api.codacy.com/project/badge/Grade/78a8256c90654c6892627f6d8bbcea14
   :target: https://www.codacy.com/gh/pgmpy/pgmpy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pgmpy/pgmpy&amp;utm_campaign=Badge_Grade

.. image:: https://img.shields.io/pypi/dm/pgmpy.svg
   :target: https://pypistats.org/packages/pgmpy

.. image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/pgmpy/pgmpy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge


.. toctree::
   :maxdepth: 2
   :hidden:

   started/base.rst
   base/base.rst
   models/base.rst
   factors/base.rst
   exact_infer/base.rst
   exact_infer/model_testing.rst
   approx_infer/base.rst
   param_estimator/base.rst
   structure_estimator/base.rst
   metrics/metrics.rst
   readwrite/base.rst
   examples.rst
   tutorial.rst

pgmpy is a pure python implementation for Bayesian Networks with a focus on
modularity and extensibility. Implementations of various alogrithms for Structure
Learning, Parameter Estimation, Approximate (Sampling Based) and Exact
inference, and Causal Inference are available.

Supported Data Types
====================

.. list-table::
   :header-rows: 1

   * -
     - Structure Learning
     - Parameter Estimation
     - Causal Inference
     - Probabilistic Inference
   * - **Discrete**
     - Yes
     - Yes
     - Yes
     - Yes
   * - **Continuous**
     - Yes (only PC)
     - No
     - Yes (partial)
     - No
   * - **Hybrid**
     - No
     - No
     - No
     - No
   * - **Time Series**
     - No
     - Yes
     - Yes (ApproximateInference)
     - Yes

Algorithms
==========

.. csv-table::
   :file: algorithms.csv
   :header-rows: 1


Example notebooks are also available at: https://github.com/pgmpy/pgmpy/tree/dev/examples

Tutorial notebooks are also available at: https://github.com/pgmpy/pgmpy_notebook

Citation
========
If you use pgmpy in your scientific work, please consider citing us:

.. code-block:: text

   Ankan, Ankur, Abinash, Panda. "pgmpy: Probabilistic Graphical Models using Python." Proceedings of the Python in Science Conference. SciPy, 2015.

Bibtex:

.. code-block:: text

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
