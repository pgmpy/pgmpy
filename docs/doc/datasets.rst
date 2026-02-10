Example Datasets
================

.. meta::
   :description: Discover and load built-in datasets in pgmpy using list_datasets and load_dataset.

The ``pgmpy.datasets`` module has two primary entry points:

- ``list_datasets`` to discover dataset names.
- ``load_dataset`` to load one dataset and its metadata.

Typical workflow:

1. Call ``list_datasets(...)`` with optional filters to find a dataset.
2. Call ``load_dataset(name)`` on one of the returned names.

list_datasets
-------------

``list_datasets(**filter_tags)`` returns a sorted ``list[str]`` of dataset names.
If no filters are provided, all available datasets are returned.

.. code-block:: python

    from pgmpy.datasets import list_datasets

    all_datasets = list_datasets()
    print(len(all_datasets))
    print(all_datasets[:5])

    # Filter by tags (exact-match filtering on tag values)
    print(list_datasets(is_discrete=True, has_ground_truth=True))
    print(list_datasets(is_continuous=True, n_variables=11))

Supported filter tags:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Tag
     - Meaning
   * - ``name``
     - Dataset identifier string.
   * - ``n_variables``
     - Number of variables (columns).
   * - ``n_samples``
     - Number of samples (rows).
   * - ``has_ground_truth``
     - Whether a ground-truth causal graph is available.
   * - ``has_expert_knowledge``
     - Whether expert knowledge constraints are available.
   * - ``has_missing_data``
     - Whether the dataset contains missing values.
   * - ``has_index_col``
     - Whether the raw file includes an index column that is removed on load.
   * - ``is_simulated``
     - Whether the dataset is simulated.
   * - ``is_interventional``
     - Whether data includes interventions.
   * - ``is_discrete``
     - Whether all variables are discrete.
   * - ``is_continuous``
     - Whether all variables are continuous.
   * - ``is_mixed``
     - Whether data contains both continuous and categorical variables.
   * - ``is_ordinal``
     - Whether ordinal variables are present.

load_dataset
------------

``load_dataset(name)`` returns a ``Dataset`` object, not just a DataFrame.
The returned object has these fields:

- ``name``: dataset name.
- ``data``: ``pandas.DataFrame`` with rows as samples and columns as variables.
- ``expert_knowledge``: ``pgmpy.estimators.ExpertKnowledge`` or ``None``.
- ``ground_truth``: ``pgmpy.base.DAG`` or ``None``.
- ``tags``: metadata dictionary for the dataset.

.. code-block:: python

    from pgmpy.datasets import load_dataset

    dataset = load_dataset("sachs_discrete")

    print(dataset.name)
    print(dataset.data.shape)
    print(dataset.tags["is_discrete"])
    print(dataset.ground_truth is not None)
    print(dataset.expert_knowledge is not None)

    # Use this DataFrame in estimators/inference workflows.
    data = dataset.data

If ``name`` does not match an available dataset, ``load_dataset`` raises
``ValueError``.

Caching behavior
----------------

On first load, dataset files are downloaded and cached under ``~/.pgmpy``.
Subsequent calls read from this local cache.

See Also
--------

- **Previous:** :doc:`simulations` -- generate synthetic data from a model
- **Next:** :doc:`example_models` -- pre-built Bayesian Networks for benchmarking
