Parameter Estimation
====================

.. meta::
   :description: Estimate CPDs for known structures using MLE, Bayesian, or EM methods.

Once a model structure is known, parameter estimation fills in the numbers for
its conditional probability distributions (CPDs).

For discrete models, maximum likelihood estimation (MLE) sets each conditional
probability from relative frequencies:

.. math::

   \hat{P}(X = x \mid Pa = p) = \frac{N(x, p)}{N(p)}

Example
-------

.. code-block:: python

    from pgmpy.datasets import load_dataset
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork

    data = load_dataset("sachs_discrete")
    model = DiscreteBayesianNetwork([("PKA", "ERK"), ("ERK", "Akt")])

    mle = MaximumLikelihoodEstimator(model, data)
    cpds = mle.get_parameters()
    model.add_cpds(*cpds)

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
