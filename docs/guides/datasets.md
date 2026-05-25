# Example Datasets

```{meta}
:description: Discover and load built-in benchmark datasets using pgmpy's dataset helpers.
```

pgmpy ships a curated collection of benchmark datasets for causal discovery and graphical
modeling. Each dataset includes not only the data table, but also metadata such as
ground-truth graphs, expert knowledge, and dataset characteristics — ready to use
without any manual download or preparation.

:::{tip}
**When to use this vs. Example Models:** *Example Datasets* provide data tables for
learning graph structures and parameters. {doc}`Example Models <example_models>` provide
pre-built graph structures (and optionally parameters) for inference, simulation, and
benchmarking.
:::

## At a Glance

- **[Unified API](#api)**: Discover datasets with `list_datasets(...)` and load them with `load_dataset(...)`.
- **[Rich Filtering](#filtering)**: Filter datasets by type, size, and available metadata before loading.
- **[Bundled Metadata](#bundled-metadata)**: Each dataset carries its ground-truth graph, expert knowledge, and tags alongside the data.

## API

The dataset API has two entry points — discover, then load:

```python
from pgmpy.datasets import list_datasets, load_dataset

matches = list_datasets(is_discrete=True, has_ground_truth=True)
dataset = load_dataset(matches[0])

print(dataset.name)
print(dataset.data.shape)
print(dataset.tags)
```

The loaded object exposes the data as a `pandas.DataFrame` and keeps all metadata on
the same object for downstream workflows.

## Filtering

`list_datasets(**filters)` narrows the catalog before loading. Supported filters include
data type (`is_discrete`, `is_continuous`, `is_mixed`), available metadata
(`has_ground_truth`, `has_expert_knowledge`), size (`n_variables`, `n_samples`), and
data origin (`is_simulated`, `is_interventional`).

## Bundled Metadata

Each loaded dataset provides structured access to:

- **`data`**: The tabular data as a `pandas.DataFrame`, ready for `estimator.fit(data)`.
- **`ground_truth`**: The true causal graph (when available), for use with supervised metrics.
- **`expert_knowledge`**: Domain constraints (when available), for use with discovery algorithms.
- **`tags`**: Dictionary of dataset properties and characteristics.

## See Also

:::{seealso}
- {doc}`Causal Discovery <causal_discovery>` — Use loaded datasets to learn causal graphs.
- {doc}`Example Models <example_models>` — Pre-built graph structures to pair with these datasets.
:::

## API Reference

For the full list of available datasets:

- {doc}`Datasets and Example Models API <../api/data>`
