Quickstart
==========

This page gives a short example for each task in pgmpy. Each section also links to the User
Guide page for that task.


.. _quickstart-causal-discovery:

Causal Discovery / Structure Learning
-------------------------------------

**User Guide:** :doc:`Causal Discovery and Structure Learning <../guides/causal_discovery>`

Learn a graph structure from data.

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

Fit the conditional distributions of a known graph structure.

.. code-block:: python

    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.parameter_estimator import DiscreteMLE
    from pgmpy.example_models import load_model

    # Generate some data to use for fitting.
    model = load_model("bnlearn/alarm")
    df = model.simulate(n_samples=1000, rng=42)

    # Create a network structure and fit data to it.
    bn_struct = DiscreteBayesianNetwork(model.edges())
    bn_struct.fit(df, estimator=DiscreteMLE())
    bn_struct.cpds


.. _quickstart-probabilistic-inference:

Probabilistic Inference
-----------------------

**User Guide:** :doc:`Probabilistic Inference <../guides/probabilistic_inference>`

Query posterior distributions from a Bayesian network.

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

Check if a causal effect is identifiable from the graph only.

.. code-block:: python

    from pgmpy.base import DAG
    from pgmpy.identification import Adjustment

    dag = DAG(
        [("X", "Y"), ("Z", "X"), ("Z", "Y")],
        roles={"exposures": "X", "outcomes": "Y"},
    )
    identified_graph, is_identified = Adjustment(variant="minimal").identify(dag)
    identified_graph.get_role("adjustment")


.. _quickstart-causal-inference:

Causal Inference
----------------

**User Guide:** :doc:`Causal Estimation <../guides/causal_estimation>`

Calculate an interventional distribution from a fitted causal model.

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

Find and load the datasets and example models.

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

Generate synthetic data from a model for tests and experiments.

.. code-block:: python

    from pgmpy.example_models import load_model

    model = load_model("bnlearn/ecoli70")
    data = model.simulate(int(1e3))
    data.head()


.. _quickstart-extensibility:

Extend pgmpy
------------

**User Guide:** :doc:`Extensibility <../guides/extensibility>`

Use the repository templates to add new datasets, models, metrics, or algorithms. pgmpy has an
extension template for each type of method. To add a new method, complete the TODO items in its
template. The templates are in https://github.com/pgmpy/pgmpy/tree/dev/devtools/extension_templates

Next Steps
----------

* :doc:`User Guide <../documentation>` -- A guide for each task
* :doc:`Examples <../examples>` -- Jupyter notebooks with longer examples
* :doc:`API Reference <../reference>` -- The documentation of all public classes and functions
