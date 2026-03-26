Causal Discovery and Structure Learning
=======================================

Discovery algorithms and expert-guided workflows for learning graph structure
from data.

Constraint and Hybrid Discovery
-------------------------------

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst

   ~pgmpy.causal_discovery.PC
   ~pgmpy.estimators.MmhcEstimator

Score-Based and Tree Search
---------------------------

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst

   ~pgmpy.causal_discovery.HillClimbSearch
   ~pgmpy.causal_discovery.GES
   ~pgmpy.estimators.TreeSearch
   ~pgmpy.estimators.ExhaustiveSearch

Expert-Guided Discovery
-----------------------

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst

   ~pgmpy.estimators.ExpertInLoop
   ~pgmpy.causal_discovery.ExpertKnowledge

.. seealso::

   :doc:`/api/ci_test` for conditional independence tests used by
   constraint-based methods.

   :doc:`/api/structure_score` for the scoring classes used by score-based
   structure search.
