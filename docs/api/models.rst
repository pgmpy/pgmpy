Models
======

Public model classes for probabilistic, causal, and structural-equation
workflows.

Directed Probabilistic Models
-----------------------------

.. autosummary::
   :toctree: generated/models
   :template: autosummary/class.rst

   ~pgmpy.models.DiscreteBayesianNetwork
   ~pgmpy.models.LinearGaussianBayesianNetwork
   ~pgmpy.models.FunctionalBayesianNetwork
   ~pgmpy.models.DynamicBayesianNetwork
   ~pgmpy.models.NaiveBayes

Structural Equation Models
--------------------------

.. autosummary::
   :toctree: generated/models
   :template: autosummary/class.rst

   ~pgmpy.models.SEMGraph
   ~pgmpy.models.SEMAlg
   ~pgmpy.models.SEM

Undirected and Derived Models
-----------------------------

.. autosummary::
   :toctree: generated/models
   :template: autosummary/class.rst

   ~pgmpy.models.DiscreteMarkovNetwork
   ~pgmpy.models.MarkovChain
   ~pgmpy.models.FactorGraph
   ~pgmpy.models.JunctionTree
   ~pgmpy.models.ClusterGraph

.. note::

   Deprecated compatibility aliases such as ``BayesianNetwork`` and
   ``MarkovNetwork`` are omitted here. Use
   ``DiscreteBayesianNetwork`` and ``DiscreteMarkovNetwork`` instead.
