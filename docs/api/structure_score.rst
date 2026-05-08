Structure Scoring
=================

Structure score classes used by score-based discovery workflows.

Base Class and Utilities
------------------------

.. autosummary::
   :toctree: generated/structure_score
   :template: autosummary/class.rst

   ~pgmpy.structure_score.BaseStructureScore

.. autosummary::
   :toctree: generated/structure_score
   :template: autosummary/function.rst

   ~pgmpy.structure_score.get_scoring_method

Discrete Scores
---------------

.. autosummary::
   :toctree: generated/structure_score
   :template: autosummary/class.rst

   ~pgmpy.structure_score.K2
   ~pgmpy.structure_score.BDeu
   ~pgmpy.structure_score.BDs
   ~pgmpy.structure_score.LogLikelihood
   ~pgmpy.structure_score.AIC
   ~pgmpy.structure_score.BIC

Gaussian Scores
---------------

.. autosummary::
   :toctree: generated/structure_score
   :template: autosummary/class.rst

   ~pgmpy.structure_score.LogLikelihoodGauss
   ~pgmpy.structure_score.AICGauss
   ~pgmpy.structure_score.BICGauss

Conditional-Gaussian Scores
---------------------------

.. autosummary::
   :toctree: generated/structure_score
   :template: autosummary/class.rst

   ~pgmpy.structure_score.LogLikelihoodCondGauss
   ~pgmpy.structure_score.AICCondGauss
   ~pgmpy.structure_score.BICCondGauss
