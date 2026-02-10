Causal Discovery and Structure Learning
========================================

Causal discovery (also known as structure learning) is the task of learning the
structure of a Directed Acyclic Graph (DAG) or its equivalence class from
observational data. The learned graph encodes conditional independence
relationships and, under certain assumptions, causal relationships between
variables.

pgmpy provides two main families of algorithms:

- **Constraint-based** methods test conditional independencies in the data to
  build the graph skeleton and orient edges (e.g., PC).
- **Score-based** methods search the space of possible graphs and optimize a
  scoring criterion (e.g., Hill-Climb, GES).

Algorithms
----------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Algorithm
     - Type
     - API Reference
   * - PC
     - Constraint-based
     - :class:`pgmpy.estimators.PC.PC`
   * - Hill-Climb Search
     - Score-based
     - :class:`pgmpy.estimators.HillClimbSearch.HillClimbSearch`
   * - Greedy Equivalence Search (GES)
     - Score-based
     - :class:`pgmpy.estimators.GES.GES`
   * - Tree Search
     - Score-based
     - :class:`pgmpy.estimators.TreeSearch.TreeSearch`
   * - Exhaustive Search
     - Score-based
     - :class:`pgmpy.estimators.ExhaustiveSearch.ExhaustiveSearch`
   * - Max-Min Hill-Climb (MMHC)
     - Hybrid
     - :class:`pgmpy.estimators.MmhcEstimator.MmhcEstimator`
   * - Expert In The Loop
     - Interactive
     - :class:`pgmpy.estimators.expert.ExpertInLoop`

Conditional Independence Tests
------------------------------

Constraint-based algorithms rely on conditional independence (CI) tests to
determine the graph structure. pgmpy provides the following CI tests:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Test
     - Description
   * - Chi-Square
     - Standard chi-squared test for discrete data
   * - G-Squared
     - G-test (log-likelihood ratio) for discrete data
   * - Log-Likelihood
     - Log-likelihood ratio test
   * - Pearson r
     - Pearson partial correlation test for continuous data
   * - Pillai Trace
     - Pillai trace test for multivariate continuous data
   * - GCM
     - Generalized Covariance Measure for nonlinear dependencies

Scoring Functions
-----------------

Score-based algorithms use scoring functions to evaluate candidate graph
structures. Available scoring functions:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Score
     - Description
   * - K2
     - K2 score for discrete Bayesian Networks
   * - BDeu
     - Bayesian Dirichlet equivalent uniform score
   * - BDs
     - Bayesian Dirichlet sparse score
   * - BIC / AIC
     - Information-theoretic scores for discrete models
   * - BICGauss / AICGauss
     - Information-theoretic scores for Gaussian models
