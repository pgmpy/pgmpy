API Reference
=============

Complete API reference for all pgmpy modules.

.. toctree::
   :hidden:
   :caption: Models

   models/base

.. toctree::
   :hidden:
   :caption: Parameterization

   factors/base

.. toctree::
   :hidden:
   :caption: Probabilistic Inference

   infer/base

.. toctree::
   :hidden:
   :caption: Causal Inference

   causal_infer/base

.. toctree::
   :hidden:
   :caption: Parameter Estimation

   param_estimator/base

.. toctree::
   :hidden:
   :caption: Causal Discovery

   structure_estimator/base

.. toctree::
   :hidden:
   :caption: Metrics

   metrics/metrics

.. toctree::
   :hidden:
   :caption: Reading/Writing

   readwrite/base

.. toctree::
   :hidden:
   :caption: Plotting

   plotting

Models
------

- :doc:`models/dag` -- Directed Acyclic Graphs
- :doc:`models/pdag` -- Partially Directed Acyclic Graphs
- :doc:`models/bayesiannetwork` -- Discrete Bayesian Networks
- :doc:`models/gaussianbn` -- Gaussian (Linear) Bayesian Networks
- :doc:`models/functionalbn` -- Functional Bayesian Networks
- :doc:`models/dbn` -- Dynamic Bayesian Networks
- :doc:`models/sem` -- Structural Equation Models
- :doc:`models/naive` -- Naive Bayes
- :doc:`models/markovnetwork` -- Markov Networks
- :doc:`models/junctiontree` -- Junction Trees
- :doc:`models/clustergraph` -- Cluster Graphs
- :doc:`models/factorgraph` -- Factor Graphs
- :doc:`models/markovchain` -- Markov Chains

Parameterization (Factors)
--------------------------

- :doc:`factors/discrete` -- Discrete CPDs and Factors
- :doc:`factors/noisyor` -- Noisy-Or Factors
- :doc:`factors/lineargauss` -- Linear Gaussian CPDs
- :doc:`factors/functional` -- Functional CPDs

Probabilistic Inference
-----------------------

**Exact:**

- :doc:`infer/ve` -- Variable Elimination
- :doc:`infer/bp` -- Belief Propagation
- :doc:`infer/bp_wmp` -- Belief Propagation (Weighted Min-Product)
- :doc:`infer/mplp` -- Max-Product Linear Programming
- :doc:`infer/dbn_infer` -- Dynamic Bayesian Network Inference

**Approximate:**

- :doc:`infer/approx_infer` -- Approximate Inference
- :doc:`infer/bn_sampling` -- Bayesian Network Sampling
- :doc:`infer/gibbs` -- Gibbs Sampling

Causal Inference
----------------

- :doc:`causal_infer/causal` -- Causal Inference Methods

Parameter Estimation
--------------------

- :doc:`param_estimator/mle` -- Maximum Likelihood Estimation
- :doc:`param_estimator/bayesian_est` -- Bayesian Estimation
- :doc:`param_estimator/em` -- Expectation Maximization
- :doc:`param_estimator/sem_estimator` -- SEM Estimation

Causal Discovery / Structure Learning
--------------------------------------

- :doc:`structure_estimator/pc` -- PC Algorithm
- :doc:`structure_estimator/hill` -- Hill-Climb Search
- :doc:`structure_estimator/ges` -- Greedy Equivalence Search
- :doc:`structure_estimator/tree` -- Tree Search
- :doc:`structure_estimator/expert` -- Expert In The Loop
- :doc:`structure_estimator/mmhc` -- Max-Min Hill-Climb
- :doc:`structure_estimator/exhaustive` -- Exhaustive Search

Metrics
-------

- :doc:`metrics/metrics` -- Metrics for Testing Models

Reading/Writing
---------------

- :doc:`readwrite/bif` -- BIF Format
- :doc:`readwrite/uai` -- UAI Format
- :doc:`readwrite/xmlbif` -- XML BIF Format
- :doc:`readwrite/xdsl` -- XDSL Format
- :doc:`readwrite/pomdpx` -- POMDPX Format
- :doc:`readwrite/xmlbelief` -- XML Belief Format

Plotting
--------

- :doc:`plotting` -- Plotting Models
