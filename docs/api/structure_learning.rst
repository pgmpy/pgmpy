Causal Discovery and Structure Learning
=======================================

Discovery algorithms and expert-guided methods that learn the graph structure from
data.

Bivariate Discovery
-------------------

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst
   :nosignatures:

   ~pgmpy.causal_discovery.ANM
   ~pgmpy.causal_discovery.IGCI

Bivariate Scores
----------------

Configurable scores for the bivariate discovery estimators.

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst
   :nosignatures:

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
   :nosignatures:

   ~pgmpy.causal_discovery.PC
   ~pgmpy.causal_discovery.MMHC
   ~pgmpy.causal_discovery.SP

Score-Based and Tree Search
---------------------------

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst
   :nosignatures:

   ~pgmpy.causal_discovery.HillClimbSearch
   ~pgmpy.causal_discovery.GES
   ~pgmpy.estimators.TreeSearch
   ~pgmpy.causal_discovery.ExhaustiveSearch
   ~pgmpy.causal_discovery.TOPIC

Order-Based Discovery
---------------------

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst
   :nosignatures:

   ~pgmpy.causal_discovery.R2Sort
   ~pgmpy.causal_discovery.VarSort

Expert-Guided Discovery
-----------------------

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst
   :nosignatures:

   ~pgmpy.estimators.ExpertInLoop
   ~pgmpy.causal_discovery.ExpertKnowledge

Stability Assessment and Bootstrapping
--------------------------------------

.. autosummary::
   :toctree: generated/structure_learning
   :template: autosummary/class.rst
   :nosignatures:

   ~pgmpy.causal_discovery.BootstrapEstimator

.. seealso::

   :doc:`/api/ci_test` for the conditional independence tests of the
   constraint-based methods.

   :doc:`/api/structure_score` for the score classes of the score-based
   structure search.
