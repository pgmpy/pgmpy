Causal Inference
================

Public classes for identification, interventional queries, and
regression-based causal effect estimation.

Graphical Identification and Inference
--------------------------------------

.. autosummary::
   :toctree: generated/causal_inference
   :template: autosummary/class.rst

   ~pgmpy.inference.CausalInference.CausalInference
   ~pgmpy.identification.adjustment.Adjustment
   ~pgmpy.identification.frontdoor.Frontdoor

Estimators
----------

.. autosummary::
   :toctree: generated/causal_inference
   :template: autosummary/class.rst

   ~pgmpy.prediction.NaiveAdjustmentRegressor.NaiveAdjustmentRegressor
   ~pgmpy.prediction.NaiveIVRegressor.NaiveIVRegressor
   ~pgmpy.prediction.DoubleMLRegressor.DoubleMLRegressor
