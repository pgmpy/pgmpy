# Probabilistic Inference

```{eval-rst}
.. meta::
   :description: Query fitted pgmpy models with the probabilistic inference algorithms.
```

Probabilistic inference calculates the distribution of query variables given observed
evidence in a graphical model. For example, take a Bayesian network for disease diagnosis.
Inference can answer this question: "What is the probability of disease X if the patient has
symptoms A and B?"

:::{note}
**Prerequisite:** You need a fitted model with parameters. Fit a model with
{doc}`Parameter Estimation <parameter_estimation>`, or load a model from
{doc}`Example Models <example_models>`.
:::

:::{tip}
**When to use this vs. Causal Estimation:** Use *Probabilistic Inference* for
observational queries that condition on evidence. For interventional queries
(do-calculus), use {doc}`Causal Estimation <causal_estimation>`.
:::

## At a Glance

::::{grid} 1 1 2 2
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

All inference algorithms use the same `query(...)` and `map_query(...)` methods.
:::

:::{grid-item-card} Exact Inference
:link: "#exact-inference"
:link-type: url
:class-card: sd-card-hover

Variable Elimination and Belief Propagation calculate the exact posterior.
:::

:::{grid-item-card} MAP Queries
:link: "#map-queries"
:link-type: url
:class-card: sd-card-hover

Find the most likely states of a set of variables.
:::

:::{grid-item-card} Approximate Inference
:link: "#approximate-inference"
:link-type: url
:class-card: sd-card-hover

Use sampling and other approximate methods for large models.
:::
::::

## API

All inference algorithms use the same steps: create the inference object from a model, then
query it.

```python
from pgmpy.example_models import load_model
from pgmpy.inference import VariableElimination  # swap in any inference engine

model = load_model("bnlearn/alarm")
infer = VariableElimination(model)

posterior = infer.query(
    variables=["HISTORY"],
    evidence={"CVP": "LOW", "PCWP": "LOW"},
)
print(posterior)
```

To use a different inference algorithm, change only the class.

## Exact Inference

Exact inference calculates the exact posterior distribution. pgmpy has two exact
algorithms:

- **Variable Elimination**: the usual choice for most queries.
- **Belief Propagation**: calibrates a junction tree one time. Repeated queries are then fast.

Use exact inference when the model is small enough for an exact calculation.

## MAP Queries

The inference algorithms also support Maximum A Posteriori (MAP) queries. A MAP query
returns the most likely joint state of the query variables:

```python
map_result = infer.map_query(variables=["HISTORY"], evidence={"CVP": "LOW"})
print(map_result)
```

## Approximate Inference

For a large network, exact inference can need too much time or memory. Then use an
approximate method. pgmpy has sampling-based inference and message-passing optimization.
It also has inference algorithms for Dynamic Bayesian Networks.

## Common Recipes

The recipes below query the ALARM network with Variable Elimination:

```python
from pgmpy.example_models import load_model
from pgmpy.inference import VariableElimination

model = load_model("bnlearn/alarm")
infer = VariableElimination(model)
```

### Posterior over a single variable

This query calculates the distribution of `HISTORY` given that `CVP` is `LOW`.

```python
infer.query(variables=["HISTORY"], evidence={"CVP": "LOW"})
```

### Joint posterior over multiple variables

With `joint=True` (the default), the result is a single distribution over all query variables.

```python
infer.query(variables=["HISTORY", "CO"], evidence={"CVP": "LOW"}, joint=True)
```

### Separate marginals instead of a joint

With `joint=False`, the result is a dictionary with one marginal distribution per variable.

```python
marginals = infer.query(variables=["HISTORY", "CO"], joint=False)
print(marginals["HISTORY"])
```

### MAP assignment

`map_query` returns the most likely joint state of the query variables given the evidence.

```python
infer.map_query(variables=["HISTORY", "CO"], evidence={"CVP": "LOW"})
```

### Switch to Belief Propagation

Belief Propagation answers the same queries. Only the inference class changes.

```python
from pgmpy.inference import BeliefPropagation

infer = BeliefPropagation(model)
infer.query(variables=["HISTORY"], evidence={"CVP": "LOW"})
```

## See Also

:::{seealso}
- {doc}`Simulations <simulations>` — Generate synthetic datasets from fitted models.
- {doc}`Causal Estimation <causal_estimation>` — Answer interventional "what if" queries with do-calculus.
:::

## API Reference

These pages list all inference algorithms and sampling methods:

- {doc}`Inference and Sampling API <../api/inference>`
- {doc}`Models API <../api/models>`
