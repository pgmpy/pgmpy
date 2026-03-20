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

    from pgmpy.example_models import load_model

    # Load a Discrete Bayesian Network and simulate data.
    discrete_bn = load_model("bnlearn/alarm")
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

    from pgmpy.example_models import load_model

    # Load an example Gaussian Bayesian Network and simulate data
    gaussian_bn = load_model("bnlearn/ecoli70")
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


Next Steps
----------
* :doc:`Examples <../examples>`
* :doc:`Contributing Guide <contributing>`
* :doc:`License <license>`
