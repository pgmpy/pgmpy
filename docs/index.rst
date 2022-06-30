.. pgmpy documentation master file, created by
   sphinx-quickstart on Tue Aug 30 18:17:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />

.. image:: logo.png
        :width: 250px
        :align: center
        :alt: logo
|

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
   :maxdepth: 3
   :hidden:
   :caption: Getting Started

   started/install.rst
   started/contributing.rst
   started/license.rst

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Base Structures

   base/base.rst

.. toctree:: :maxdepth: 3
   :hidden:
   :caption: Models

   models/bayesiannetwork.rst
   models/dbn.rst
   models/sem.rst
   models/naive.rst
   models/noisyor.rst
   models/markovnetwork.rst
   models/junctiontree.rst
   models/clustergraph.rst
   models/factorgraph.rst
   models/markovchain.rst

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Parameterization

   factors/discrete.rst
   factors/continuous.rst
   factors/discretize.rst

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Exact Inference

   exact_infer/ve.rst
   exact_infer/bp.rst
   exact_infer/causal.rst
   exact_infer/mplp.rst
   exact_infer/dbn_infer.rst
   exact_infer/model_testing.rst

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Approximate Inference

   approx_infer/approx_infer.rst
   approx_infer/bn_sampling.rst
   approx_infer/gibbs.rst

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Parameter Estimation

   param_estimator/mle.rst
   param_estimator/bayesian_est.rst
   param_estimator/em.rst
   param_estimator/sem_estimator.rst

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Structure Learning

   structure_estimator/pc.rst
   structure_estimator/hill.rst
   structure_estimator/tree.rst
   structure_estimator/mmhc.rst
   structure_estimator/exhaustive.rst

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Model Testing

   metrics/metrics.rst

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Input/Output

   readwrite/bif.rst
   readwrite/uai.rst
   readwrite/xmlbif.rst
   readwrite/pomdpx.rst
   readwrite/xmlbelief.rst

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
   * - Discrete
     - Yes
     - Yes
     - Yes
     - Yes
   * - Continuous
     - Yes (only PC)
     - No
     - Yes (partial)
     - No
   * - Hybrid
     - No
     - No
     - No
     - No
   * - Time Series
     - No
     - Yes
     - No
     - Yes

Algorithms
==========

.. csv-table::
   :file: algorithms.csv
   :header-rows: 1

.. toctree::
   :maxdepth: 1
   :caption: Example Notebooks
   :numbered:

   examples/Earthquake.ipynb
   examples/Monty Hall Problem.ipynb
   examples/Creating a Discrete Bayesian Network.ipynb
   examples/Inference in Discrete Bayesian Networks.ipynb
   examples/Causal Games.ipynb
   examples/Causal Inference.ipynb
   examples/Learning Parameters in Discrete Bayesian Networks.ipynb
   examples/Structure Learning in Bayesian Networks.ipynb
   examples/Structure Learning with Chow-Liu.ipynb
   examples/Structure Learning with TAN.ipynb
   examples/Simulating Data.ipynb
   examples/Extending pgmpy.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Tutorial Notebooks
   :numbered:

   detailed_notebooks/1. Introduction to Probabilistic Graphical Models.ipynb
   detailed_notebooks/2. Bayesian Networks.ipynb
   detailed_notebooks/3. Causal Bayesian Networks.ipynb
   detailed_notebooks/4. Markov Models.ipynb
   detailed_notebooks/5. Exact Inference in Graphical Models.ipynb
   detailed_notebooks/6. Approximate Inference in Graphical Models.ipynb
   detailed_notebooks/7. Parameterizing with Continuous Variables.ipynb
   detailed_notebooks/8. Sampling Algorithms.ipynb
   detailed_notebooks/9. Reading and Writing from pgmpy file formats.ipynb
   detailed_notebooks/10. Learning Bayesian Networks from Data.ipynb
   detailed_notebooks/11. A Bayesian Network to model the influence of energy consumption on greenhouse gases in Italy.ipynb


All example notebooks are also available at: https://github.com/pgmpy/pgmpy/tree/dev/examples
All tutorial notebooks are also available at: https://github.com/pgmpy/pgmpy_notebook

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
