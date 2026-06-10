# Defining a Custom Model

```{meta}
:description: Build custom pgmpy models and attach the right CPD or factor types for your workflow.
```

When you already know the structure of a model, you can define the graph and its
parameters directly in pgmpy. This workflow gives you full control over nodes, edges,
and CPDs, and supports several model families beyond standard discrete Bayesian networks.

## At a Glance

- **[Unified API](#api)**: Create a model, add CPDs, and validate — the same pattern across all model families.
- **[Multiple Model Families](#model-families)**: Discrete, continuous, functional, dynamic, and undirected models.
- **[Model Validation](#model-validation)**: Verify consistency of structure and parameters before downstream use.

## API

The common pattern is: choose a model class, define the structure, add CPDs, and validate:

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

Once valid, the model works with inference, simulation, and the rest of the pgmpy stack.

## Model Families

pgmpy supports several model families, each paired with a matching CPD or factor type.
These include discrete Bayesian networks with tabular CPDs, continuous networks with
linear Gaussian CPDs, functional models with arbitrary CPDs, dynamic Bayesian networks
for temporal processes, undirected Markov networks, and structural equation models.

Each family uses the same create → add CPDs → validate pattern shown above. For example,
a continuous model:

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

`model.check_model()` verifies that the graph structure and attached CPDs are consistent
— correct parent sets, probability tables that sum to 1, matching cardinalities, etc.
Always validate before using a model downstream.

## See Also

:::{seealso}
- {doc}`Parameter Estimation <parameter_estimation>` — Learn CPDs from data instead of defining them manually.
- {doc}`Probabilistic Inference <probabilistic_inference>` — Query the model once it is built and validated.
:::

## API Reference

For the full list of model classes and factor types:

- {doc}`Models API <../api/models>`
- {doc}`Factors and CPDs API <../api/factors>`
- {doc}`Graph Classes API <../api/base>`
