# Example Models

```{meta}
:description: Discover and load built-in example models using pgmpy's example-model helpers.
```

pgmpy ships a registry of ready-made Bayesian network models from standard repositories.
These range from structure-only causal graphs to fully parameterized networks that can be
queried and simulated immediately — useful for benchmarking, demonstrations, and exploring
the API.

:::{tip}
**When to use this vs. Example Datasets:** *Example Models* provide pre-built graph
structures (and optionally parameters) ready for inference and simulation.
{doc}`Example Datasets <datasets>` provide data tables for learning structures and
parameters from scratch.
:::

## At a Glance

- **[Unified API](#api)**: Discover models with `list_models(...)` and load them with `load_model(...)`.
- **[Rich Filtering](#filtering)**: Filter models by type, parameterization, and size before loading.
- **[Multiple Model Families](#model-families)**: Discrete, continuous, and structure-only models from bnlearn, bnrep, and dagitty repositories.

## API

The example-model API mirrors the dataset API — discover, then load:

```python
from pgmpy.example_models import list_models, load_model

names = list_models(is_parameterized=True, is_discrete=True)
model = load_model(names[0])

print(model)
print(len(model.nodes()), len(model.edges()))
```

The returned object type depends on the model family (e.g., `DiscreteBayesianNetwork`,
`LinearGaussianBayesianNetwork`, or `DAG`).

## Filtering

`list_models(**filters)` narrows the registry before loading. Supported filters include
`is_parameterized`, `is_discrete`, `is_continuous`, `is_hybrid`, `n_nodes`, and `n_edges`.

## Model Families

- **Discrete parameterized**: Classic networks with full CPD tables, ready for inference
  and simulation.
- **Continuous**: Networks with linear Gaussian parameters.
- **Structure-only**: Causal graphs without parameters, useful for identification and
  causal reasoning tasks.

Loaded models integrate directly with the rest of the pgmpy stack — inference, simulation,
parameter estimation, and plotting all work out of the box.

## See Also

:::{seealso}
- {doc}`Probabilistic Inference <probabilistic_inference>` — Query loaded parameterized models.
- {doc}`Simulations <simulations>` — Sample data from loaded parameterized models.
- {doc}`Example Datasets <datasets>` — Companion data tables for benchmarking.
:::

## API Reference

For the full list of available models:

- {doc}`Datasets and Example Models API <../api/data>`
- {doc}`Models API <../api/models>`
