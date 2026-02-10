Defining a Custom Model
=======================

.. meta::
   :description: Define custom graphical models and CPD types in pgmpy to build Bayesian and Markov models.

pgmpy lets you define different graphical model families and their
corresponding factor or CPD types.

For a DAG model, the joint distribution factorizes as a product of local
conditionals over parents:

.. math::

   P(X_1, \ldots, X_n) = \prod_{i=1}^n P(X_i \mid Pa_i)

Example
-------

.. code-block:: python

    from pgmpy.datasets import load_dataset
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork

    data = load_dataset("sachs_discrete")
    model = DiscreteBayesianNetwork([("PKA", "ERK"), ("ERK", "Akt")])
    fitted = model.fit(data, estimator=MaximumLikelihoodEstimator)
    print(fitted.get_cpds())

Model Types
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Model
     - CPD Type
     - API Reference
   * - Bayesian Network (Discrete)
     - TabularCPD
     - :class:`~pgmpy.models.BayesianNetwork.BayesianNetwork`
   * - Linear Gaussian BN
     - LinearGaussianCPD
     - :class:`~pgmpy.models.LinearGaussianBayesianNetwork.LinearGaussianBayesianNetwork`
   * - Functional BN
     - FunctionalCPD
     - :class:`~pgmpy.models.FunctionalBayesianNetwork.FunctionalBayesianNetwork`
   * - Dynamic BN
     - TabularCPD
     - :class:`~pgmpy.models.DynamicBayesianNetwork.DynamicBayesianNetwork`
   * - Naive Bayes
     - TabularCPD
     - :class:`~pgmpy.models.NaiveBayes.NaiveBayes`
   * - Markov Network
     - DiscreteFactor
     - :class:`~pgmpy.models.MarkovNetwork.MarkovNetwork`
   * - Factor Graph
     - DiscreteFactor
     - :class:`~pgmpy.models.FactorGraph.FactorGraph`
   * - Junction Tree
     - DiscreteFactor
     - :class:`~pgmpy.models.JunctionTree.JunctionTree`
   * - Cluster Graph
     - DiscreteFactor
     - :class:`~pgmpy.models.ClusterGraph.ClusterGraph`
   * - Structural Equation Model
     - --
     - :class:`~pgmpy.models.SEM.SEM`
   * - Markov Chain
     - --
     - :class:`~pgmpy.models.MarkovChain.MarkovChain`

Factor / CPD Types
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Factor
     - API Reference
   * - TabularCPD
     - :class:`~pgmpy.factors.discrete.CPD.TabularCPD`
   * - DiscreteFactor
     - :class:`~pgmpy.factors.discrete.DiscreteFactor.DiscreteFactor`
   * - NoisyOrCPD
     - :class:`~pgmpy.factors.discrete.NoisyOR.NoisyOrCPD`
   * - LinearGaussianCPD
     - :class:`~pgmpy.factors.continuous.LinearGaussianCPD.LinearGaussianCPD`
   * - FunctionalCPD
     - :class:`~pgmpy.factors.hybrid.FunctionalCPD.FunctionalCPD`
