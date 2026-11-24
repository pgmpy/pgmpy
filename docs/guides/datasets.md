# Example Datasets

```{eval-rst}
.. meta::
   :description: Find and load the benchmark datasets in pgmpy.
```

pgmpy has a collection of benchmark datasets for causal discovery and graphical models.
Each dataset contains the data table and its metadata, for example the ground-truth graph,
expert knowledge, and tags. pgmpy downloads the data when you load a dataset.

:::{tip}
**When to use this vs. Example Models:** *Example Datasets* contain data tables. Use them
to learn graph structures and parameters. {doc}`Example Models <example_models>` contain
graph structures, and some also contain parameters. Use them for inference, simulation, and
benchmarks.
:::

## At a Glance

::::{grid} 1 1 2 2
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

Find datasets with `list_datasets(...)` and load them with `load_dataset(...)`.
:::

:::{grid-item-card} Filtering
:link: "#filtering"
:link-type: url
:class-card: sd-card-hover

Filter the datasets by data type, size, and metadata before you load them.
:::

:::{grid-item-card} Bundled Metadata
:link: "#bundled-metadata"
:link-type: url
:class-card: sd-card-hover

Each dataset contains its ground-truth graph, expert knowledge, and tags together with the data.
:::

:::{grid-item-card} Available Datasets
:link: "#available-datasets"
:link-type: url
:class-card: sd-card-hover

A table of all datasets with their size, data type, and metadata.
:::
::::

## API

The dataset API has two functions. One function finds datasets, and the other loads them:

```python
from pgmpy.datasets import list_datasets, load_dataset

matches = list_datasets(is_discrete=True, has_ground_truth=True)
dataset = load_dataset(matches[0])

print(dataset.name)
print(dataset.data.shape)
print(dataset.tags)
```

The loaded object contains the data as a `pandas.DataFrame` and all the metadata of the
dataset.

## Filtering

`list_datasets(**filters)` returns only the datasets that match the filters. You can
filter on:

- data type: `is_discrete`, `is_continuous`, `is_mixed`.
- metadata: `has_ground_truth`, `has_expert_knowledge`.
- size: `n_variables`, `n_samples`.
- data origin: `is_simulated`, `is_interventional`.

## Bundled Metadata

Each loaded dataset has these attributes:

- **`data`**: The data table as a `pandas.DataFrame`. Give it directly to `estimator.fit(data)`.
- **`ground_truth`**: The true causal graph, if the dataset has one. Use it with supervised metrics.
- **`expert_knowledge`**: Domain constraints, if the dataset has them. Use them with discovery
  algorithms.
- **`tags`**: A dictionary with the properties of the dataset.

## Available Datasets

This table shows all datasets that `list_datasets()` returns, with the tags that you can
filter on. `tubingen` is a collection of 108 cause-effect pairs. To load one pair, use
`load_dataset("tubingen/<pair_id>")`, with a `<pair_id>` from 1 to 108. A simulated dataset
gives the number of samples that you request.

```{eval-rst}
.. pgmpy-catalog:: datasets
```

## See Also

:::{seealso}
- {doc}`Causal Discovery <causal_discovery>` — Use loaded datasets to learn causal graphs.
- {doc}`Example Models <example_models>` — Graph structures to use with these datasets.
:::

## API Reference

- {doc}`Datasets and Example Models API <../api/data>`
