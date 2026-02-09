Example Models
==============

The ``pgmpy.example_models`` module provides a curated set of example models
that can be loaded directly for experimentation, benchmarking, and tutorials.

Available Methods
-----------------

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - :func:`pgmpy.example_models.list_models`
     - Returns a sorted list of available model names. Supports filtering using
       model tags such as ``is_discrete``, ``is_continuous``, and
       ``is_parameterized``.
   * - :func:`pgmpy.example_models.load_model`
     - Loads a model by name and returns the corresponding model object.

Examples
--------

List models
"""""""""""

.. code-block:: python

    from pgmpy.example_models import list_models

    # List all available example models.
    all_models = list_models()

    # Filter by model tags.
    discrete_models = list_models(is_discrete=True)
    dag_only_models = list_models(is_parameterized=False)

Load models
"""""""""""

.. code-block:: python

    from pgmpy.example_models import load_model

    # Load a parameterized discrete Bayesian network.
    alarm = load_model("alarm")

    # Load a DAG without parameters.
    confounding = load_model("confounding")

    # Load a parameterized continuous Bayesian network.
    arth150 = load_model("arth150")

API Reference
-------------

.. automodule:: pgmpy.example_models
   :members:
   :exclude-members: _BaseExampleModel
