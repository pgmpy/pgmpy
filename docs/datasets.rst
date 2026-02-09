Datasets
========

The ``pgmpy.datasets`` module provides curated datasets for structure learning,
causal discovery, and benchmarking.

Available Methods
-----------------

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - :func:`pgmpy.datasets.list_datasets`
     - Returns a sorted list of available dataset names. Supports filtering
       using dataset tags such as ``is_discrete``, ``is_continuous``, and
       ``has_ground_truth``.
   * - :func:`pgmpy.datasets.load_dataset`
     - Loads a dataset by name and returns a ``Dataset`` object with data,
       metadata tags, and optional expert knowledge and ground truth.

Examples
--------

List datasets
"""""""""""""

.. code-block:: python

    from pgmpy.datasets import list_datasets

    # List all available datasets.
    all_datasets = list_datasets()

    # Filter by dataset tags.
    continuous_datasets = list_datasets(is_continuous=True)
    datasets_with_ground_truth = list_datasets(has_ground_truth=True)

Load a dataset
""""""""""""""

.. code-block:: python

    from pgmpy.datasets import load_dataset

    dataset = load_dataset("sachs_mixed")

    # Access the dataframe and metadata.
    data = dataset.data
    tags = dataset.tags

    # Optional metadata, depending on dataset tags.
    expert_knowledge = dataset.expert_knowledge
    ground_truth = dataset.ground_truth

Supported Datasets
------------------

This table is generated from dataset metadata during each docs build.

.. csv-table::
   :file: datasets.csv
   :header-rows: 1

API Reference
-------------

.. automodule:: pgmpy.datasets
   :members:
   :exclude-members: _BaseDataset
