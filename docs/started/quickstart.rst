Quickstart
==========

Task-oriented examples to help you get started with pgmpy.

Learn Structure from Data
-------------------------

Discover the causal graph from observational data using constraint-based
or score-based algorithms.

.. code-block:: python

    from pgmpy.utils import get_example_model
    from pgmpy.estimators import PC

    model = get_example_model("alarm")
    df = model.simulate(n_samples=1000)

    learned_dag = PC(data=df).estimate(ci_test="chi_square", return_type="dag")

Estimate Parameters
-------------------

Fit conditional probability distributions to a known graph structure
using Maximum Likelihood or Bayesian estimation.

.. code-block:: python

    from pgmpy.utils import get_example_model

    model = get_example_model("alarm")
    df = model.simulate(n_samples=1000)

    # Fit parameters to the learned structure
    from pgmpy.models import BayesianNetwork

    bn = BayesianNetwork(model.edges())
    bn.fit(df)
    bn.get_cpds("HISTORY")

Run Probabilistic Inference
---------------------------

Query the posterior probability of variables given evidence using
exact or approximate inference.

.. code-block:: python

    from pgmpy.utils import get_example_model
    from pgmpy.inference import VariableElimination

    model = get_example_model("alarm")
    infer = VariableElimination(model)

    result = infer.query(
        variables=["HISTORY"],
        evidence={"CVP": "LOW", "PCWP": "LOW"},
    )
    print(result)

Perform Causal Inference
------------------------

Estimate causal effects using do-calculus, backdoor adjustment,
or frontdoor adjustment.

.. code-block:: python

    from pgmpy.utils import get_example_model
    from pgmpy.inference import CausalInference

    model = get_example_model("alarm")
    infer = CausalInference(model)

    # Compute the causal effect of intervention
    result = infer.query(
        variables=["HISTORY"],
        do={"LVEDVOLUME": "LOW"},
    )
    print(result)

Simulate Data from a Model
---------------------------

Generate synthetic datasets from an existing Bayesian Network
for testing and experimentation.

.. code-block:: python

    from pgmpy.utils import get_example_model

    model = get_example_model("alarm")
    df = model.simulate(n_samples=500)
    print(df.head())

Build a Custom Model
--------------------

Define a Bayesian Network from scratch by specifying the graph
structure and conditional probability distributions.

.. code-block:: python

    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD

    bn = BayesianNetwork([("D", "G"), ("I", "G"), ("G", "L")])

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

    bn.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l)
    bn.check_model()

Next Steps
----------

* :doc:`Examples <../examples>` -- Jupyter notebooks with detailed walkthroughs
* :doc:`API Reference <../api>` -- Full module documentation
