# Probabilistic Inference

```{meta}
:description: Query fitted pgmpy models using the unified probabilistic inference APIs.
```

Probabilistic inference computes the distribution over query variables given observed
evidence in a graphical model. For example, given a Bayesian network modeling disease
diagnosis, inference answers questions like "What is the probability of disease X given
symptoms A and B?".

:::{note}
**Prerequisite:** This guide assumes you have a fitted model with parameters. Obtain one
from {doc}`Parameter Estimation <parameter_estimation>` or load a ready-made one from
{doc}`Example Models <example_models>`.
:::

:::{tip}
**When to use this vs. Causal Estimation:** Use *Probabilistic Inference* for
observational queries that condition on evidence. For interventional queries that model
interventions (do-calculus), see {doc}`Causal Estimation <causal_estimation>`.
:::

## At a Glance

- **[Unified API](#api)**: All inference engines share a `query(...)` / `map_query(...)` interface.
- **[Exact Inference](#exact-inference)**: Variable Elimination and Belief Propagation for precise posterior computation.
- **[MAP Queries](#map-queries)**: Find the most likely assignment for a set of variables.
- **[Approximate Inference](#approximate-inference)**: Sampling-based and other approximate methods for large models.

## API

All inference engines follow the same pattern — instantiate with a model, then query:

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

Switching inference engines requires only changing the class.

## Exact Inference

Exact inference computes the precise posterior distribution. pgmpy provides Variable
Elimination (the default choice for most queries) and Belief Propagation (efficient for
repeated queries through junction-tree reasoning). These are best when the model is small
enough for exact computation to be tractable.

## MAP Queries

In addition to full posterior distributions, inference engines support Maximum A Posteriori
queries that return the single most likely assignment:

```python
map_result = infer.map_query(variables=["HISTORY"], evidence={"CVP": "LOW"})
print(map_result)
```

## Approximate Inference

When exact inference is too expensive for large networks, pgmpy provides approximate
methods including sampling-based inference and message-passing optimization. There is
also specialized support for inference on Dynamic Bayesian Networks.

## Common Recipes

```python
from pgmpy.example_models import load_model
from pgmpy.inference import VariableElimination

model = load_model("bnlearn/alarm")
infer = VariableElimination(model)
```

**Posterior over a single variable:**
```python
infer.query(variables=["HISTORY"], evidence={"CVP": "LOW"})
```

**Joint posterior over multiple variables:**
```python
infer.query(variables=["HISTORY", "CO"], evidence={"CVP": "LOW"}, joint=True)
```

**Separate marginals instead of joint:**
```python
marginals = infer.query(variables=["HISTORY", "CO"], joint=False)
print(marginals["HISTORY"])
```

**MAP assignment:**
```python
infer.map_query(variables=["HISTORY", "CO"], evidence={"CVP": "LOW"})
```

**Switch to Belief Propagation:**
```python
from pgmpy.inference import BeliefPropagation

infer = BeliefPropagation(model)
infer.query(variables=["HISTORY"], evidence={"CVP": "LOW"})
```

## See Also

:::{seealso}
- {doc}`Simulations <simulations>` — Generate synthetic datasets from fitted models.
- {doc}`Causal Estimation <causal_estimation>` — Answer interventional "what if" queries using do-calculus.
:::

## API Reference

For the full list of inference algorithms and sampling methods:

- {doc}`Inference and Sampling API <../api/inference>`
- {doc}`Models API <../api/models>`
