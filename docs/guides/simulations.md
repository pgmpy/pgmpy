# Simulations

```{eval-rst}
.. meta::
   :description: Generate synthetic data from fitted pgmpy models.
```

Simulation draws samples from the joint distribution of a graphical model. The graph and
its parameters define this distribution. Use simulated data to:

- compare algorithms with a known ground truth.
- make controlled experimental datasets.
- test the behavior of a model under interventions.

:::{note}
**Prerequisite:** You need a fitted model with parameters. Fit a model with
{doc}`Parameter Estimation <parameter_estimation>`, or load a model from
{doc}`Example Models <example_models>`.
:::

## At a Glance

::::{grid} 1 1 2 3
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

One `model.simulate(...)` method does all types of simulation.
:::

:::{grid-item-card} Interventional Simulation
:link: "#interventional-simulation"
:link-type: url
:class-card: sd-card-hover

Generate data under hard do-interventions.
:::

:::{grid-item-card} Conditional Simulation
:link: "#conditional-simulation"
:link-type: url
:class-card: sd-card-hover

Draw samples conditioned on observed evidence.
:::

:::{grid-item-card} Soft Evidence
:link: "#soft-evidence"
:link-type: url
:class-card: sd-card-hover

Use virtual evidence and virtual interventions.
:::

:::{grid-item-card} Missing Data Generation
:link: "#missing-data-generation"
:link-type: url
:class-card: sd-card-hover

Make datasets with a controlled pattern of missing values.
:::
::::

## API

The main method is `model.simulate(...)`:

```python
from pgmpy.example_models import load_model

model = load_model("bnlearn/alarm")
samples = model.simulate(n_samples=1000, rng=42, show_progress=False)

print(samples.head())
```

For more control of the sampling algorithm, use the classes in `pgmpy.sampling`.

## Interventional Simulation

The `do` argument sets variables to fixed values. This is Pearl's do-operator, and it
gives interventional data:

```python
samples = model.simulate(n_samples=1000, do={"CVP": "LOW"}, show_progress=False)
```

## Conditional Simulation

The `evidence` argument gives samples conditioned on observed values:

```python
samples = model.simulate(n_samples=1000, evidence={"CVP": "LOW", "PCWP": "HIGH"}, show_progress=False)
```

## Soft Evidence

The `virtual_evidence` and `virtual_intervention` arguments change the probabilities of
the states of a variable. They do not set the variable to a fixed value:

```python
from pgmpy.example_models import load_model
from pgmpy.factors.discrete import TabularCPD

model = load_model("bnlearn/alarm")

soft = TabularCPD("CVP", 3, [[0.7], [0.2], [0.1]])
samples = model.simulate(n_samples=100, virtual_evidence=[soft], rng=42, show_progress=False)
print(samples["CVP"].value_counts(normalize=True))
```

## Missing Data Generation

The `missing_prob` argument makes datasets with controlled missing values. Use these
datasets to test algorithms for incomplete data. The `include_latents`, `partial_samples`,
and `return_full` arguments control what the output contains.

## Common Recipes

The recipes below sample from the ALARM network:

```python
from pgmpy.example_models import load_model

model = load_model("bnlearn/alarm")
```

### Forward sampling

Draw samples from the joint distribution of the model. Set `rng` to get the same samples
each time.

```python
samples = model.simulate(n_samples=1000, rng=42, show_progress=False)
```

### Interventional data (do-operator)

`do` sets `CVP` to `LOW` in every sample. It also removes the effect of the parents of `CVP`.

```python
samples = model.simulate(n_samples=1000, do={"CVP": "LOW"}, show_progress=False)
```

### Conditional sampling (evidence)

`evidence` gives samples from the distribution conditioned on `CVP = LOW`.

```python
samples = model.simulate(n_samples=1000, evidence={"CVP": "LOW"}, show_progress=False)
```

### Generate data with missing values

`missing_prob` takes a `TabularCPD`. Its variable name is the name of the variable followed
by `*`. State 1 means missing. This recipe makes 10% of the `HISTORY` values missing.

```python
from pgmpy.factors.discrete import TabularCPD

missing = TabularCPD("HISTORY*", 2, [[0.9], [0.1]])
samples = model.simulate(n_samples=1000, missing_prob=missing, show_progress=False)
```

### Include latent variables in the output

The samples do not contain latent variables, unless you set `include_latents=True`.

```python
model.latents = {"LVFAILURE"}
samples = model.simulate(n_samples=1000, include_latents=True, show_progress=False)
```

## See Also

:::{seealso}
- {doc}`Probabilistic Inference <probabilistic_inference>` — Calculate probabilities from the model without samples.
:::

## API Reference

These pages list all simulation and sampling methods:

- {doc}`Inference and Sampling API <../api/inference>`
- {doc}`Models API <../api/models>`
