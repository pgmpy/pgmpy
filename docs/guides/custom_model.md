# Defining a Custom Model

```{eval-rst}
.. meta::
   :description: Make custom pgmpy models and add the correct CPD or factor types.
```

If you know the structure of a model, you can define the graph and its parameters in pgmpy
yourself. You then have full control of the nodes, edges, and CPDs. This works for discrete
Bayesian networks and for other model families.

## At a Glance

::::{grid} 1 1 3 3
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

Create a model, add the CPDs, and validate it. The steps are the same for all model families.
:::

:::{grid-item-card} Model Families
:link: "#model-families"
:link-type: url
:class-card: sd-card-hover

Discrete, continuous, functional, dynamic, and undirected models.
:::

:::{grid-item-card} Model Validation
:link: "#model-validation"
:link-type: url
:class-card: sd-card-hover

Make sure that the structure and the parameters agree before you use the model.
:::
::::

## API

The steps are: select a model class, define the structure, add the CPDs, and validate the
model:

```python
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])

cpd_a = TabularCPD("A", 2, [[0.6], [0.4]])
cpd_b = TabularCPD("B", 2, [[0.9, 0.2], [0.1, 0.8]], evidence=["A"], evidence_card=[2])
cpd_c = TabularCPD("C", 2, [[0.3, 0.7], [0.7, 0.3]], evidence=["A"], evidence_card=[2])

model.add_cpds(cpd_a, cpd_b, cpd_c)
print(model.check_model())
```

You can then use the model with inference, simulation, and all other parts of pgmpy.

## Model Families

pgmpy has these model families. Each family uses its own CPD or factor type:

- discrete Bayesian networks with tabular CPDs.
- continuous networks with linear Gaussian CPDs.
- functional models with arbitrary CPDs.
- dynamic Bayesian networks for temporal processes.
- undirected Markov networks.
- structural equation models.

All families use the same steps as the example above: create the model, add the CPDs, and
validate it. This example makes a continuous model:

```python
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD

model = LinearGaussianBayesianNetwork([("X", "Y"), ("Z", "Y")])

cpd_x = LinearGaussianCPD("X", beta=[0.0], std=1.0)
cpd_z = LinearGaussianCPD("Z", beta=[0.0], std=1.0)
cpd_y = LinearGaussianCPD("Y", beta=[0.2, 0.5, 0.3], std=1.0, evidence=["X", "Z"])

model.add_cpds(cpd_x, cpd_z, cpd_y)
print(model.check_model())
```

## Model Validation

`model.check_model()` makes sure that the graph structure and the CPDs agree. For example,
it checks the parent sets, the cardinalities, and that each probability table sums to 1.
Always validate a model before you use it.

## See Also

:::{seealso}
- {doc}`Parameter Estimation <parameter_estimation>` — Learn the CPDs from data.
- {doc}`Probabilistic Inference <probabilistic_inference>` — Query the model after you create and validate it.
:::

## API Reference

These pages list all model classes and factor types:

- {doc}`Models API <../api/models>`
- {doc}`Factors and CPDs API <../api/factors>`
- {doc}`Graph Classes API <../api/base>`
