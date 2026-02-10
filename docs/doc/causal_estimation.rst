Causal Estimation
=================

.. meta::
   :description: Estimate causal effects from data after identification using do-calculus and regression-based estimators.

Causal estimation quantifies how much changing a variable changes an outcome.

More precisely, it estimates causal effects such as the average treatment
effect (ATE) from observed data using an identified adjustment set:

.. math::

   ATE = E[Y \mid do(X=1)] - E[Y \mid do(X=0)]

Example
-------

.. code-block:: python

    from pgmpy.datasets import load_dataset
    from pgmpy.inference import CausalInference
    from pgmpy.models import DiscreteBayesianNetwork

    data = load_dataset("sachs_discrete")
    dag = DiscreteBayesianNetwork(
        [
            ("PKA", "ERK"),
            ("ERK", "Akt"),
            ("PKA", "Akt"),
        ]
    )
    ci = CausalInference(dag)
    ate = ci.estimate_ate("PKA", "Akt", data)
    print(ate)

When to use which
-----------------

- **CausalInference** -- Use with a fully specified causal graph to compute
  interventional distributions via do-calculus.
- **Naive Adjustment Regressor** -- Simple backdoor adjustment with a
  regression model. Good starting point for continuous outcomes.
- **Naive IV Regressor** -- Instrumental variable regression when there is
  unmeasured confounding but a valid instrument is available.
- **Double ML Regressor** -- Doubly-robust estimation using machine learning
  models. Best for high-dimensional settings or when the outcome model may be
  misspecified.

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
