Causal Discovery and Structure Learning
=======================================

Discovery algorithms and expert-guided workflows for learning graph structure
from data.

Bivariate Discovery
-------------------

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst

   ~pgmpy.causal_discovery.ANM
   ~pgmpy.causal_discovery.IGCI

Bivariate Scores
----------------

Configurable scoring methods used by the bivariate discovery estimators.

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst

   ~pgmpy.causal_discovery.bivariate_scores.BaseBivariateScore
   ~pgmpy.causal_discovery.bivariate_scores.IndependenceScore
   ~pgmpy.causal_discovery.bivariate_scores.EntropyScore
   ~pgmpy.causal_discovery.bivariate_scores.EntropyDifferenceScore
   ~pgmpy.causal_discovery.bivariate_scores.GaussScore
   ~pgmpy.causal_discovery.bivariate_scores.SlopeScore

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/function.rst

   ~pgmpy.causal_discovery.bivariate_scores.get_bivariate_score

Constraint and Hybrid Discovery
-------------------------------

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst

   ~pgmpy.causal_discovery.PC
   ~pgmpy.estimators.MmhcEstimator
   ~pgmpy.causal_discovery.SP

Score-Based and Tree Search
---------------------------

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst

   ~pgmpy.causal_discovery.HillClimbSearch
   ~pgmpy.causal_discovery.GES
   ~pgmpy.estimators.TreeSearch
   ~pgmpy.estimators.ExhaustiveSearch
   ~pgmpy.causal_discovery.TOPIC

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
