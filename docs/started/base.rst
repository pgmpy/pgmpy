Getting Started
===============

Install
-------
pgmpy supports Python >= 3.10. For installation through pypi:

.. code-block:: bash

        pip install pgmpy

For installation through anaconda, use the command:

.. code-block:: bash

        conda install -c conda-forge pgmpy

For installing the latest `dev` branch from GitHub, use the command:

.. code-block:: bash

        pip install git+https://github.com/pgmpy/pgmpy.git@dev

Quickstart
----------

Discrete Bayesian Network
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pgmpy.utils import get_example_model

    # Load a Discrete Bayesian Network and simulate data.
    discrete_bn = get_example_model("alarm")
    alarm_df = discrete_bn.simulate(n_samples=100)

    # Learn a network from simulated data.
    from pgmpy.estimators import PC

    dag = PC(data=alarm_df).estimate(ci_test="chi_square", return_type="dag")

    # Learn the parameters from the data.
    dag_fitted = dag.fit(alarm_df)
    dag_fitted.get_cpds()


Gaussian Bayesian Network
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pgmpy.utils import get_example_model

    # Load an example Gaussian Bayesian Network and simulate data
    gaussian_bn = get_example_model("ecoli70")
    ecoli_df = gaussian_bn.simulate(n_samples=100)

    # Learn the network from simulated data.
    from pgmpy.estimators import PC

    dag = PC(data=ecoli_df).estimate(ci_test="pearsonr", return_type="dag")

    # Learn the parameters from the data.
    from pgmpy.models import LinearGaussianBayesianNetwork

    gaussian_bn = LinearGaussianBayesianNetwork(dag.edges())
    dag_fitted = gaussian_bn.fit(ecoli_df)
    dag_fitted.get_cpds()

    # Drop a column and predict using the learned model.
    evidence_df = ecoli_df.drop(columns=["ftsJ"], axis=1)
    pred_ftsJ = dag_fitted.predict(evidence_df)

Mixture Data with arbitrary distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pgmpy.global_vars import config

    config.set_backend("torch")

    import pyro.distributions as dist

    from pgmpy.models import FunctionalBayesianNetwork
    from pgmpy.factors.hybrid import FunctionalCPD

    # Create a Bayesian Network with mixture of discrete and continuous variables.
    func_bn = FunctionalBayesianNetwork(
        [
            ("x1", "w"),
            ("x2", "w"),
            ("x1", "y"),
            ("x2", "y"),
            ("w", "y"),
            ("y", "z"),
            ("w", "z"),
            ("y", "c"),
            ("w", "c"),
        ]
    )

    # Define the Functional CPDs for each node and add them to the model.
    cpd_x1 = FunctionalCPD("x1", fn=lambda _: dist.Normal(0.0, 1.0))
    cpd_x2 = FunctionalCPD("x2", fn=lambda _: dist.Normal(0.5, 1.2))

    # Continuous mediator: w = 0.7*x1 - 0.3*x2 + ε
    cpd_w = FunctionalCPD(
        "w",
        fn=lambda parents: dist.Normal(0.7 * parents["x1"] - 0.3 * parents["x2"], 0.5),
        parents=["x1", "x2"],
    )

    # Bernoulli target with logistic link: y ~ Bernoulli(sigmoid(-0.7 + 1.5*x1 + 0.8*x2 + 1.2*w))
    cpd_y = FunctionalCPD(
        "y",
        fn=lambda parents: dist.Bernoulli(
            logits=(
                -0.7 + 1.5 * parents["x1"] + 0.8 * parents["x2"] + 1.2 * parents["w"]
            )
        ),
        parents=["x1", "x2", "w"],
    )

    # Downstream Bernoulli influenced by y and w
    cpd_z = FunctionalCPD(
        "z",
        fn=lambda parents: dist.Bernoulli(
            logits=(-1.2 + 0.8 * parents["y"] + 0.2 * parents["w"])
        ),
        parents=["y", "w"],
    )

    # Continuous outcome depending on y and w: c = 0.2 + 0.5*y + 0.3*w + ε
    cpd_c = FunctionalCPD(
        "c",
        fn=lambda parents: dist.Normal(
            0.2 + 0.5 * parents["y"] + 0.3 * parents["w"], 0.7
        ),
        parents=["y", "w"],
    )

    func_bn.add_cpds(cpd_x1, cpd_x2, cpd_w, cpd_y, cpd_z, cpd_c)
    func_bn.check_model()

    # Simulate data from the model
    df_func = func_bn.simulate(n_samples=1000, seed=123)

    # For learning and inference in Functional Bayesian Networks, please refer to the example notebook: https://github.com/pgmpy/pgmpy/blob/dev/examples/Functional_Bayesian_Network_Tutorial.ipynb


Benchmark Datasets (ALARM and Asia)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pgmpy provides two classic Bayesian Network benchmarks via on-demand loaders:

- **ALARM**: A 37-node benchmark for intensive-care monitoring and probabilistic inference evaluation.
- **Asia**: An 8-node benchmark (also called Lung Cancer) often used in teaching and quick benchmarking.

.. code-block:: python

    from pgmpy.datasets import load_alarm, load_asia, get_benchmark_metadata

    # Sample benchmark data.
    alarm_data = load_alarm(n_samples=1000, sample_id=0, random_state=42)
    asia_data = load_asia(n_samples=1000)

    # Optionally return the benchmark model as well.
    alarm_data_2, alarm_model = load_alarm(n_samples=1000, return_model=True)

    # Access citation and license metadata.
    meta = get_benchmark_metadata("alarm")
    print(meta["citation"])
    print(meta["license"])

The loader downloads model files on first use and stores them in
``PGMPY_DATA_HOME/benchmark_datasets`` (``PGMPY_DATA_HOME`` defaults to ``~/.pgmpy``).
To refresh the cached model file, set ``force_download=True``.

These benchmark assets include citation and license information in
``get_benchmark_metadata``.


Next Steps
----------
* :doc:`Examples <../examples>`
* :doc:`Contributing Guide <contributing>`
* :doc:`License <license>`
