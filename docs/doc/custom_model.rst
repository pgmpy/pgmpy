Defining a Custom Model
=======================

pgmpy supports several types of graphical models. Each model type pairs with
specific factor (CPD) types to define the joint probability distribution.

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

Example: Discrete Bayesian Network
-----------------------------------

.. code-block:: python

    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD

    # Define structure
    model = BayesianNetwork([("D", "G"), ("I", "G"), ("G", "L")])

    # Define CPDs
    cpd_d = TabularCPD("D", 2, [[0.6], [0.4]])
    cpd_i = TabularCPD("I", 2, [[0.7], [0.3]])
    cpd_g = TabularCPD(
        "G",
        3,
        [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        evidence=["D", "I"],
        evidence_card=[2, 2],
    )
    cpd_l = TabularCPD(
        "L",
        2,
        [[0.1, 0.4, 0.99], [0.9, 0.6, 0.01]],
        evidence=["G"],
        evidence_card=[3],
    )

    model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l)
    model.check_model()  # Returns True if valid

Example: Linear Gaussian Bayesian Network
------------------------------------------

.. code-block:: python

    from pgmpy.models import LinearGaussianBayesianNetwork
    from pgmpy.factors.continuous import LinearGaussianCPD

    model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])

    cpd_x = LinearGaussianCPD("X", [0.5], 1.0)
    cpd_y = LinearGaussianCPD("Y", [0.2, 0.8], 0.5, ["X"])
    cpd_z = LinearGaussianCPD("Z", [-0.3, 1.2], 0.3, ["Y"])

    model.add_cpds(cpd_x, cpd_y, cpd_z)
    model.check_model()
