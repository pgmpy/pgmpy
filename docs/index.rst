.. pgmpy documentation master file, created by
   sphinx-quickstart on Tue Aug 30 18:17:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />

.. image:: logo.png
        :width: 250px

|

.. image:: https://github.com/pgmpy/pgmpy/actions/workflows/ci.yml/badge.svg?branch=dev
   :target: https://github.com/pgmpy/pgmpy/actions?query=branch%3Adev

.. image:: https://codecov.io/gh/pgmpy/pgmpy/branch/dev/graph/badge.svg
   :target: https://codecov.io/gh/pgmpy/pgmpy

.. image:: https://api.codacy.com/project/badge/Grade/78a8256c90654c6892627f6d8bbcea14
   :target: https://www.codacy.com/gh/pgmpy/pgmpy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pgmpy/pgmpy&amp;utm_campaign=Badge_Grade

|br|

.. image:: https://img.shields.io/pypi/dm/pgmpy.svg
   :target: https://pypistats.org/packages/pgmpy

.. image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/pgmpy/pgmpy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat
   :target: http://pgmpy.org/pgmpy-benchmarks/

|

Basic Examples: For a complete list check: https://github.com/pgmpy/pgmpy/tree/dev/examples

1. `Defining a Discrete Bayesian Network <https://github.com/pgmpy/pgmpy/blob/dev/examples/Creating%20a%20Discrete%20Bayesian%20Network.ipynb>`_
2. `Statistical Inference in Discrete Bayesian Network <https://github.com/pgmpy/pgmpy/blob/dev/examples/Inference%20in%20Discrete%20Bayesian%20Networks.ipynb>`_
3. `Causal Inference <https://github.com/pgmpy/pgmpy/blob/dev/examples/Causal%20Games.ipynb>`_
4. `Learning Discrete Bayesian Networks from Data <https://github.com/pgmpy/pgmpy/blob/dev/examples/Learning%20Parameters%20in%20Discrete%20Bayesian%20Networks.ipynb>`_
5. `Learning Bayesian Networks structures from Data <https://github.com/pgmpy/pgmpy/blob/dev/examples/Structure%20Learning%20in%20Bayesian%20Networks.ipynb>`_

Detailed Notebooks:

1. `Introduction to Probabilistic Graphical Models <https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/1.%20Introduction%20to%20Probabilistic%20Graphical%20Models.ipynb>`_
2. `Bayesian Networks <https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/2.%20Bayesian%20Networks.ipynb>`_
3. `Markov Models <https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/3.%20Markov%20Models.ipynb>`_
4. `Exact Inference in Graphical Models <https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/4.%20Exact%20Inference%20in%20Graphical%20Models.ipynb>`_
5. `Approximate Inference in Graphical Models <https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/5.%20Approximate%20Inference%20in%20Graphical%20Models.ipynb>`_
6. `Parameterizing with continuous variables <https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/6.%20Parameterizing%20with%20Continuous%20Variables.ipynb>`_
7. `Sampling Algorithms <https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/7.%20Sampling%20Algorithms.ipynb>`_
8. `Learning Bayesian Networks from data <https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/9.%20Learning%20Bayesian%20Networks%20from%20Data.ipynb>`_
9. `Reading and writing files using pgmpy <https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/8.%20Reading%20and%20Writing%20from%20pgmpy%20file%20formats.ipynb>`_

Documentation:

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

.. toctree::
   :maxdepth: 3
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
   structrue_estimator/tree.rst
   structure_estimator/mmhc.rst
   structure_estimator/exhaustive.rst

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Input/Output

   readwrite/bif.rst
   readwrite/uai.rst
   readwrite/xmlbif.rst
   readwrite/pomdpx.rst
   readwrite/xmlbelief.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

