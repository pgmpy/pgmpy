Quickstart
==========

Task-oriented examples to help you get started with pgmpy.

The sections below provide a minimal example for various supported tasks and a link to the corresponding User Guide page for the full workflow.


.. _quickstart-causal-discovery:

Causal Discovery / Structure Learning
-------------------------------------

**User Guide:** :doc:`Causal Discovery and Structure Learning <../guides/causal_discovery>`

Learn a graph structure directly from data.

.. code-block:: python

    from pgmpy.datasets import load_dataset
    from pgmpy.causal_discovery import PC

    dataset = load_dataset("sachs_discrete")
    est = PC(ci_test="chi_square", return_type="dag")
    est.fit(dataset.data)
    print(est.causal_graph_)


.. _quickstart-parameter-estimation:

Parameter Estimation
--------------------

**User Guide:** :doc:`Parameter Estimation <../guides/parameter_estimation>`

Fit conditional distributions for a known graph structure.

.. code-block:: python

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.parameter_estimator import DiscreteMLE
    from pgmpy.example_models import load_model

    # Generate some data to use for fitting.
    model = load_model("bnlearn/alarm")
    df = model.simulate(n_samples=1000, seed=42)

    # Create a network structure and fit data to it.
    bn_struct = DiscreteBayesianNetwork(model.edges())
    bn_struct.fit(df, estimator=DiscreteMLE())
    bn_struct.cpds


.. _quickstart-probabilistic-inference:

Probabilistic Inference
-----------------------

**User Guide:** :doc:`Probabilistic Inference <../guides/probabilistic_inference>`

Query posterior distributions from a Bayesian Network.

.. code-block:: python

    from pgmpy.example_models import load_model
    from pgmpy.inference import VariableElimination

    model = load_model("bnlearn/alarm")
    infer = VariableElimination(model)

    result = infer.query(
        variables=["HISTORY"],
        evidence={"CVP": "LOW", "PCWP": "LOW"},
    )
    print(result)


.. _quickstart-causal-identification:

Causal Identification
---------------------

**User Guide:** :doc:`Causal Identification <../guides/causal_identification>`

Check whether a causal effect is identifiable from the graph alone.

.. code-block:: python

    from pgmpy.base import DAG
    from pgmpy.identification import Adjustment

    dag = DAG(
        [("X", "Y"), ("Z", "X"), ("Z", "Y")],
        roles={"exposures": "X", "outcomes": "Y"},
    )
    identified_graph, is_identified = Adjustment(variant="minimal").identify(dag)
    identified.get_role("adjustment")


.. _quickstart-causal-inference:

Causal Inference
----------------

**User Guide:** :doc:`Causal Estimation <../guides/causal_estimation>`

Estimate a causal effect from data once you have a causal graph.

.. code-block:: python

    from pgmpy.example_models import load_model
    from pgmpy.inference import CausalInference

    model = load_model("bnlearn/sachs")
    infer = CausalInference(model)
    infer.query(variables=["Akt"], do={"PKC": "LOW"})


.. _quickstart-example-data-models:

Example Datasets and Models
---------------------------

**User Guides:** :doc:`Example Datasets <../guides/datasets>` | :doc:`Example Models <../guides/example_models>`

Discover built-in datasets and example models.

.. code-block:: python

    from pgmpy.datasets import list_datasets, load_dataset
    from pgmpy.example_models import list_models, load_model

    print(list_datasets(is_discrete=True, has_ground_truth=True)[:3])
    dataset = load_dataset("sachs_discrete")
    print(dataset.name, dataset.data.shape)

    print(list_models()[:3])
    model = load_model("bnlearn/alarm")
    print(len(model.nodes()), len(model.edges()))


.. _quickstart-simulations:

Simulations
-----------

**User Guide:** :doc:`Simulations <../guides/simulations>`

Generate synthetic data from a model for testing and experimentation.

.. code-block:: python

    from pgmpy.example_models import load_model

    model = load_model("bnlearn/ecoli70")
    data = model.simulate(int(1e3))
    data.head()


.. _quickstart-extensibility:

Extend pgmpy
------------

**User Guide:** :doc:`Extensibility <../guides/extensibility>`

Implement new datasets, models, metrics, or algorithms from the repository templates. pgmpy provides extension templates
for each class of method. To add a new method, you can just follow the TODOs listed in the extension template for that
method. All of our extension templates can be found at: https://github.com/pgmpy/pgmpy/tree/dev/devtools/extension_templates

Next Steps
----------

* :doc:`User Guide <../documentation>` -- Workflow-oriented documentation for each task area
* :doc:`Examples <../examples>` -- Jupyter notebooks with longer walkthroughs
* :doc:`API Reference <../reference>` -- Full public API documentation
