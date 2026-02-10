Parameter Estimation
====================

Once the structure of a graphical model is known, the next step is to estimate
the parameters -- the conditional probability distributions (CPDs) at each node.
pgmpy supports several estimation methods depending on the data type, model
type, and whether data is fully observed or contains missing values.

Algorithms
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - API Reference
   * - Maximum Likelihood Estimation (MLE)
     - :class:`pgmpy.estimators.MLE.MaximumLikelihoodEstimator`
   * - Bayesian Estimation
     - :class:`pgmpy.estimators.BayesianEstimator.BayesianEstimator`
   * - Expectation Maximization (EM)
     - :class:`pgmpy.estimators.EM.ExpectationMaximization`
   * - SEM Estimator
     - :class:`pgmpy.estimators.SEMEstimator.SEMEstimator`
   * - IV Estimator
     - :class:`pgmpy.estimators.SEMEstimator.IVEstimator`

When to use which
-----------------

- **MLE** -- The default choice when data is fully observed. Computes CPDs
  directly from frequency counts (discrete) or regression (continuous).
- **Bayesian Estimation** -- Useful with small datasets where MLE may overfit.
  Incorporates prior knowledge via Dirichlet priors (e.g., BDeu).
- **EM** -- Required when the data contains missing values. Iterates between
  imputing missing data and re-estimating parameters.
- **SEM / IV Estimator** -- For Structural Equation Models with continuous
  variables and linear relationships.
