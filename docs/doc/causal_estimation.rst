Causal Estimation
=================

Causal estimation computes the magnitude of a causal effect from data once
identifiability has been established. pgmpy provides both graphical model-based
and semi-parametric estimation methods.

The graphical model-based approach uses do-calculus to simulate interventions on
a fitted Bayesian Network. The semi-parametric methods use regression-based
estimators that are robust to model misspecification.

Algorithms
----------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Algorithm
     - API Reference
   * - CausalInference (do-calculus)
     - :class:`pgmpy.inference.CausalInference.CausalInference`

Semi-parametric Estimators
--------------------------

These methods combine graphical structure with flexible regression models for
heterogeneous treatment effect estimation.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Algorithm
     - API Reference
   * - Naive Adjustment Regressor
     - :class:`pgmpy.prediction.NaiveAdjustmentRegressor.NaiveAdjustmentRegressor`
   * - Naive IV Regressor
     - :class:`pgmpy.prediction.NaiveIVRegressor.NaiveIVRegressor`
   * - Double ML Regressor
     - :class:`pgmpy.prediction.DoubleMLRegressor.DoubleMLRegressor`

When to use which
-----------------

- **CausalInference** -- Use with a fully specified and fitted Bayesian Network
  to compute interventional distributions via do-calculus.
- **Naive Adjustment Regressor** -- Simple backdoor adjustment with a
  regression model. Good starting point for continuous outcomes.
- **Naive IV Regressor** -- Instrumental variable regression when there is
  unmeasured confounding but a valid instrument is available.
- **Double ML Regressor** -- Doubly-robust estimation using machine learning
  models. Best for high-dimensional settings or when the outcome model may be
  misspecified.
