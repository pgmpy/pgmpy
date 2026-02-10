Causal Discovery / Structure Learning
=====================================

pgmpy provides causal discovery algorithms with an sklearn-compatible API. All algorithms
follow the standard `fit` / `predict` pattern and can be used with sklearn's model selection
utilities.

Quick Start
-----------

.. code-block:: python

    from pgmpy.utils import get_example_model
    from pgmpy.causal_discovery import PC

    model = get_example_model("asia")
    df = model.simulate(n_samples=1000, seed=42)

    pc = PC(ci_test="chi_square", significance_level=0.01)
    pc.fit(df)

    print(pc.causal_graph_.edges())
    print(pc.adjacency_matrix_)

Using Expert Knowledge
----------------------

Expert knowledge can be incorporated to constrain the search space:

.. code-block:: python

    from pgmpy.causal_discovery import HillClimbSearch
    from pgmpy.estimators import ExpertKnowledge

    expert = ExpertKnowledge(
        forbidden_edges=[("smoke", "asia")],
        required_edges=[("smoke", "lung")],
    )

    hc = HillClimbSearch(scoring_method="bic-d", expert_knowledge=expert)
    hc.fit(df)

Available Algorithms
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Algorithm
     - Type
     - Description
   * - :doc:`pc`
     - Constraint
     - PC algorithm using conditional independence tests
   * - :doc:`hill`
     - Score
     - Hill climbing search with tabu list
   * - :doc:`ges`
     - Score
     - Greedy Equivalence Search over CPDAG space
   * - :doc:`tree`
     - Score
     - Tree structure learning (Chow-Liu, TAN)
   * - :doc:`mmhc`
     - Hybrid
     - Max-Min Hill Climbing
   * - :doc:`exhaustive`
     - Score
     - Exhaustive search (small graphs only)
   * - :doc:`expert`
     - -
     - Expert knowledge specification

Algorithm Details
-----------------

.. toctree::
   :maxdepth: 2

   pc.rst
   hill.rst
   ges.rst
   tree.rst
   expert.rst
   mmhc.rst
   exhaustive.rst
