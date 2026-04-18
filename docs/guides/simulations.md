# Simulations

```{meta}
:description: Generate synthetic data from fitted pgmpy models using the simulation APIs.
```

Simulation generates synthetic samples from a graphical model by sampling from the
joint distribution defined by the graph and its parameters. This is useful for
benchmarking against known ground truth, creating controlled experimental datasets,
and testing model behavior under interventions.

:::{note}
**Prerequisite:** Simulation requires a fitted model with parameters. Fit one with
{doc}`Parameter Estimation <parameter_estimation>` or load a ready-made one from
{doc}`Example Models <example_models>`.
:::

## At a Glance

- **[Unified API](#api)**: A single `model.simulate(...)` call handles all simulation scenarios.
- **[Interventional Simulation](#interventional-simulation)**: Generate data under hard do-interventions.
- **[Conditional Simulation](#conditional-simulation)**: Sample conditioned on observed evidence.
- **[Soft Evidence](#soft-evidence)**: Support for virtual evidence and virtual interventions.
- **[Missing Data Generation](#missing-data-generation)**: Create datasets with controlled missingness patterns.

## API

The primary entry point is `model.simulate(...)`:

```python
from pgmpy.example_models import load_model

model = load_model("bnlearn/alarm")
samples = model.simulate(n_samples=1000, seed=42, show_progress=False)

print(samples.head())
```

For more control over the sampling algorithm, use the classes in `pgmpy.sampling`
directly.

## Interventional Simulation

The `do` parameter fixes specific variables to given values, implementing Pearl's
do-operator to generate interventional data:

```python
samples = model.simulate(n_samples=1000, do={"CVP": "LOW"}, show_progress=False)
```

## Conditional Simulation

The `evidence` parameter generates samples conditioned on observed values:

```python
samples = model.simulate(
    n_samples=1000, evidence={"CVP": "LOW", "PCWP": "HIGH"}, show_progress=False
)
```

## Soft Evidence

`virtual_evidence` and `virtual_intervention` parameters support soft constraints where
variables are influenced probabilistically rather than set to fixed values:

```python
from pgmpy.example_models import load_model
from pgmpy.factors.discrete import TabularCPD

model = load_model("bnlearn/alarm")

soft = TabularCPD("CVP", 3, [[0.7], [0.2], [0.1]])
samples = model.simulate(
    n_samples=100, virtual_evidence=[soft], seed=42, show_progress=False
)
print(samples["CVP"].value_counts(normalize=True))
```

## Missing Data Generation

The `missing_prob` parameter generates datasets with controlled missingness, useful for
testing algorithms that handle incomplete data. Additional parameters like
`include_latents`, `partial_samples`, and `return_full` control what appears in the
output.

## Common Recipes

```python
from pgmpy.example_models import load_model

model = load_model("bnlearn/alarm")
```

**Basic forward sampling:**
```python
samples = model.simulate(n_samples=1000, seed=42, show_progress=False)
```

**Interventional data (do-operator):**
```python
samples = model.simulate(n_samples=1000, do={"CVP": "LOW"}, show_progress=False)
```

**Conditional sampling (evidence):**
```python
samples = model.simulate(n_samples=1000, evidence={"CVP": "LOW"}, show_progress=False)
```

**Generate data with 10% missing values:**
```python
samples = model.simulate(n_samples=1000, missing_prob=0.1, show_progress=False)
```

**Include latent variables in output:**
```python
samples = model.simulate(n_samples=1000, include_latents=True, show_progress=False)
```

## See Also

:::{seealso}
- {doc}`Probabilistic Inference <probabilistic_inference>` — Query the model analytically instead of sampling.
:::

## API Reference

For the full list of simulation and sampling methods:

- {doc}`Inference and Sampling API <../api/inference>`
- {doc}`Models API <../api/models>`
