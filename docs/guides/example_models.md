# Example Models

```{eval-rst}
.. meta::
   :description: Find and load the example models in pgmpy.
```

pgmpy has a collection of Bayesian network models from standard repositories. Some models
contain only the graph structure. Other models also contain all parameters, so you can query
them and draw samples from them. Use the models for benchmarks, demonstrations, and to learn
the API.

:::{tip}
**When to use this vs. Example Datasets:** *Example Models* contain graph structures, and
some also contain parameters. Use them for inference and simulation.
{doc}`Example Datasets <datasets>` contain data tables. Use them to learn structures and
parameters from data.
:::

## At a Glance

::::{grid} 1 1 2 2
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

Find models with `list_models(...)` and load them with `load_model(...)`.
:::

:::{grid-item-card} Filtering
:link: "#filtering"
:link-type: url
:class-card: sd-card-hover

Filter the models by type, parameters, and size before you load them.
:::

:::{grid-item-card} Model Families
:link: "#model-families"
:link-type: url
:class-card: sd-card-hover

Discrete, continuous, and structure-only models from the bnlearn, bnRep, and DAGitty
repositories.
:::

:::{grid-item-card} Available Models
:link: "#available-models"
:link-type: url
:class-card: sd-card-hover

A table of all models with their size and type.
:::
::::

## API

The example-model API is the same as the dataset API. One function finds models, and the
other loads them:

```python
from pgmpy.example_models import list_models, load_model

names = list_models(is_parameterized=True, is_discrete=True)
model = load_model(names[0])

print(model)
print(len(model.nodes()), len(model.edges()))
```

The type of the returned object depends on the model family, for example
`DiscreteBayesianNetwork`, `LinearGaussianBayesianNetwork`, or `DAG`.

## Filtering

`list_models(**filters)` returns only the models that match the filters. You can filter on
`is_parameterized`, `is_discrete`, `is_continuous`, `is_hybrid`, `n_nodes`, and `n_edges`.

## Model Families

- **Discrete parameterized**: Networks with all CPD tables. Use them for inference and
  simulation.
- **Continuous**: Networks with linear Gaussian parameters.
- **Structure-only**: Causal graphs without parameters. Use them for identification and
  causal reasoning.

You can use a loaded model with all other parts of pgmpy, for example inference,
simulation, parameter estimation, and plots.

## Available Models

This table shows all models that `list_models()` returns. The prefix of each name shows the
source repository:

- `bnlearn/`: the [bnlearn Bayesian network repository](https://www.bnlearn.com/bnrepository/).
- `bnrep/`: [bnRep](https://github.com/manueleleonelli/bnRep).
- `dagitty/`: the example DAGs of [DAGitty](https://dagitty.net).

```{eval-rst}
.. pgmpy-catalog:: example_models
```

## See Also

:::{seealso}
- {doc}`Probabilistic Inference <probabilistic_inference>` — Query loaded parameterized models.
- {doc}`Simulations <simulations>` — Draw samples from loaded parameterized models.
- {doc}`Example Datasets <datasets>` — Data tables for benchmarks.
:::

## API Reference

- {doc}`Datasets and Example Models API <../api/data>`
- {doc}`Models API <../api/models>`
