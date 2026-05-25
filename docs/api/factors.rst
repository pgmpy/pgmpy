Factors and CPDs
================

Factor, CPD, and factor-container classes used to parameterize pgmpy models.

Discrete Factors and CPDs
-------------------------

.. autosummary::
   :toctree: generated/factors
   :template: autosummary/class.rst

   ~pgmpy.factors.discrete.TabularCPD
   ~pgmpy.factors.discrete.DiscreteFactor
   ~pgmpy.factors.discrete.JointProbabilityDistribution
   ~pgmpy.factors.discrete.NoisyORCPD

Continuous and Hybrid CPDs
--------------------------

.. autosummary::
   :toctree: generated/factors
   :template: autosummary/class.rst

   ~pgmpy.factors.continuous.LinearGaussianCPD
   ~pgmpy.factors.hybrid.FunctionalCPD

Factor Containers and Utilities
-------------------------------

.. autosummary::
   :toctree: generated/factors
   :template: autosummary/class.rst

   ~pgmpy.factors.FactorSet
   ~pgmpy.factors.FactorDict

.. autosummary::
   :toctree: generated/factors
   :template: autosummary/function.rst

   ~pgmpy.factors.factor_product
   ~pgmpy.factors.factor_divide
   ~pgmpy.factors.factor_sum_product
   ~pgmpy.factors.factorset_product
   ~pgmpy.factors.factorset_divide
