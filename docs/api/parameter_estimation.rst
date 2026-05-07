Parameter Estimation
====================

Estimator classes for fitting model parameters and related marginal
representations from data.

Estimator Base Classes
----------------------

.. autosummary::
   :toctree: generated/parameter_estimation
   :template: autosummary/class.rst

   ~pgmpy.parameter_estimator.BaseParameterEstimator
   ~pgmpy.parameter_estimator.DiscreteParameterEstimator
   ~pgmpy.parameter_estimator.GaussianParameterEstimator

Discrete Bayesian Network Estimators
------------------------------------

Estimators for :class:`~pgmpy.models.DiscreteBayesianNetwork` models.

.. autosummary::
   :toctree: generated/parameter_estimation
   :template: autosummary/class.rst

   ~pgmpy.parameter_estimator.DiscreteMLE
   ~pgmpy.parameter_estimator.DiscreteBayesianEstimator
   ~pgmpy.parameter_estimator.DiscreteEM

Linear Gaussian Bayesian Network Estimators
-------------------------------------------

Estimators for :class:`~pgmpy.models.LinearGaussianBayesianNetwork` models.

.. autosummary::
   :toctree: generated/parameter_estimation
   :template: autosummary/class.rst

   ~pgmpy.parameter_estimator.LinearGaussianMLE

Structural Equation Model Estimators
------------------------------------

.. autosummary::
   :toctree: generated/parameter_estimation
   :template: autosummary/class.rst

   ~pgmpy.estimators.SEMEstimator
   ~pgmpy.estimators.IVEstimator

Marginal and Graphical Model Fitting
------------------------------------

.. autosummary::
   :toctree: generated/parameter_estimation
   :template: autosummary/class.rst

   ~pgmpy.estimators.MirrorDescentEstimator
