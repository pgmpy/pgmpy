Causal Discovery and Structure Learning
=======================================

.. meta::
   :description: Learn causal graphs from data using constraint- and score-based structure learning in pgmpy.

Causal discovery (structure learning) finds which variables influence others
by learning a directed graph from data.

In more precise terms, we seek a directed acyclic graph (DAG) :math:`G` over
variables :math:`X` that either satisfies conditional independencies like
:math:`X \perp Y \mid Z` (constraint-based) or maximizes a score :math:`S(G; D)`
(score-based) for data :math:`D`:

.. math::

   G^* = \arg\max_G S(G; D)

Example
-------

.. code-block:: python

    from pgmpy.datasets import load_dataset
    from pgmpy.estimators import HillClimbSearch, BIC

    data = load_dataset("sachs_discrete")
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=BIC(data))
    print(model.edges())

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

See Also
--------

- **Examples:** :doc:`Structure Learning <../examples/Structure_Learning>` | :doc:`Chow-Liu Tree <../examples/Structure_Learning_Chow_Liu>` | :doc:`TAN <../examples/Structure_Learning_TAN>` | :doc:`Expert Knowledge <../examples/Expert_Knowledge>`
- **API Reference:** :doc:`Causal Discovery API <../structure_estimator/base>`
- **Previous:** :doc:`custom_model` -- define a model from scratch
- **Next:** :doc:`parameter_estimation` -- estimate CPDs for the learned structure
