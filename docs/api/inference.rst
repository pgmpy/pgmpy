Inference and Sampling
======================

Core Inference Classes
----------------------

.. autosummary::
   :toctree: generated/inference
   :template: autosummary/class.rst

   ~pgmpy.inference.Inference
   ~pgmpy.inference.VariableElimination
   ~pgmpy.inference.BeliefPropagation
   ~pgmpy.inference.BeliefPropagationWithMessagePassing
   ~pgmpy.inference.Mplp
   ~pgmpy.inference.DBNInference

Approximate Inference and Sampling
----------------------------------

.. autosummary::
   :toctree: generated/inference
   :template: autosummary/class.rst

   ~pgmpy.inference.ApproxInference
   ~pgmpy.sampling.BayesianModelInference
   ~pgmpy.sampling.BayesianModelSampling
   ~pgmpy.sampling.GibbsSampling

Elimination-Order Heuristics
----------------------------

.. autosummary::
   :toctree: generated/inference
   :template: autosummary/class.rst

   ~pgmpy.inference.EliminationOrder.BaseEliminationOrder
   ~pgmpy.inference.EliminationOrder.WeightedMinFill
   ~pgmpy.inference.EliminationOrder.MinNeighbors
   ~pgmpy.inference.EliminationOrder.MinWeight
   ~pgmpy.inference.EliminationOrder.MinFill
