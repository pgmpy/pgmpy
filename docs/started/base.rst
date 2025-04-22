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
   discrete_bn = get_example_model('alarm')
   alarm_df = discrete_bn.simulate(n_samples=100)

   # Learn a network from simulated data.
   from pgmpy.estimators import PC
   dag = PC(data=alarm_df).estimate(ci_test='chi_square', return_type='dag')

   # Learn the parameters from the data.
   dag_fitted = dag.fit(alarm_df)
   dag_fitted.get_cpds()

Gaussian Bayesian Network
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Load an example Gaussian Bayesian Network and simulate data
   gaussian_bn = get_example_model('ecoli70')
   ecoli_df = gaussian_bn.simulate(n_samples=100)

   # Learn the network from simulated data.
   from pgmpy.estimators import PC
   dag = PC(data=ecoli_df).estimate(ci_test='pearsonr', return_type='dag')

   # Learn the parameters from the data.
   gaussian_bn = LinearGausianBayesianNetwork(dag.edges())
   dag_fitted = gaussian_bn.fit(ecoli_df)
   dag_fitted.get_cpds()


Next Steps
----------
* :doc:`Examples <../examples>`
* :doc:`Contributing Guide <contributing>`
* :doc:`License <license>`
