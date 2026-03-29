# Parameter Estimation

```{meta}
:description: Estimate model parameters from data using pgmpy's unified parameter-estimation APIs.
```

Once the structure of a graphical model is known, the next step is estimating its
numerical parameters from data. For a Bayesian network, this means learning the
conditional probability distribution (CPD) for every node given its parents.

:::{note}
**Prerequisite:** This guide assumes you already have a graph structure, either from
{doc}`Causal Discovery <causal_discovery>` or built manually with
{doc}`Defining a Custom Model <custom_model>`.
:::

## At a Glance

- **[Unified API](#api)**: A single `model.fit(data, estimator=...)` call handles all estimation methods.
- **[Prior-Based Smoothing](#prior-based-smoothing)**: Bayesian estimation with Dirichlet priors for sparse data.
- **[Missing Data and Latent Variables](#missing-data-and-latent-variables)**: Expectation Maximization for incomplete observations.
- **[Parallel Estimation](#parallel-estimation)**: Speed up fitting for larger models with `n_jobs`.

## API

The recommended entry point is `model.fit(...)`. Provide the graph, the data, and the
estimator class:

```python
from pgmpy.example_models import load_model
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork

reference = load_model("bnlearn/alarm")
data = reference.simulate(n_samples=1000, seed=42, show_progress=False)

model = DiscreteBayesianNetwork(reference.edges())
model.fit(data, estimator=MaximumLikelihoodEstimator)

print(model.get_cpds("HISTORY"))
```

Switching the estimation method only requires changing the `estimator` argument.
For finer control, you can instantiate estimator classes directly and call their
parameter-generation methods before adding the CPDs to the model.

## Prior-Based Smoothing

When data is sparse, maximum likelihood estimates can be unreliable. The Bayesian
estimator adds Dirichlet priors (pseudo-counts and equivalent sample size) to smooth
the estimated distributions and incorporate prior beliefs.

## Missing Data and Latent Variables

When the dataset has missing values or the model contains latent variables, simple
counting is not enough. The Expectation Maximization estimator handles this through
iterative estimation, alternating between imputing missing values and updating parameters.

## Parallel Estimation

The `n_jobs` parameter in `model.fit(...)` parallelizes parameter estimation across nodes,
which can speed up fitting for larger models. The `state_names` parameter lets you
specify the supported states explicitly when some states may not appear in small datasets.

## See Also

:::{seealso}
- {doc}`Probabilistic Inference <probabilistic_inference>` — Query the fitted model.
- {doc}`Simulations <simulations>` — Sample synthetic data from the fitted model.
:::

## API Reference

For the full list of supported estimators:

- {doc}`Parameter Estimation API <../api/parameter_estimation>`
- {doc}`Models API <../api/models>`
- {doc}`Factors and CPDs API <../api/factors>`
