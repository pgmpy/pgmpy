# Parameter Estimation

```{eval-rst}
.. meta::
   :description: Estimate the parameters of a model from data with pgmpy.
```

After you have the structure of a graphical model, estimate its parameters from data. For
a Bayesian network, the parameters are the conditional probability distribution (CPD) of
each node given its parents.

:::{note}
**Prerequisite:** You need a graph structure. Learn it with
{doc}`Causal Discovery <causal_discovery>`, or make it yourself with
{doc}`Defining a Custom Model <custom_model>`.
:::

## At a Glance

::::{grid} 1 1 2 2
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

One `model.fit(data, estimator=...)` method does all types of estimation.
:::

:::{grid-item-card} Prior-Based Smoothing
:link: "#prior-based-smoothing"
:link-type: url
:class-card: sd-card-hover

Use Bayesian estimation with Dirichlet priors for sparse data.
:::

:::{grid-item-card} Missing Data and Latent Variables
:link: "#missing-data-and-latent-variables"
:link-type: url
:class-card: sd-card-hover

Use Expectation Maximization for incomplete data.
:::

:::{grid-item-card} Parallel Estimation
:link: "#parallel-estimation"
:link-type: url
:class-card: sd-card-hover

Use `n_jobs` to fit large models faster.
:::
::::

## API

Make a model from the graph. Then call `model.fit(...)` with the data and an estimator
instance:

```python
from pgmpy.example_models import load_model
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE

reference = load_model("bnlearn/alarm")
data = reference.simulate(n_samples=1000, rng=42, show_progress=False)

model = DiscreteBayesianNetwork(reference.edges())
model.fit(data, estimator=DiscreteMLE())

print(model.get_cpds("HISTORY"))
```

To use a different estimation method, change only the estimator instance. For more
control, create an estimator and call its `fit(...)` method yourself. Then add the CPDs to
the model.

## Prior-Based Smoothing

If the data is sparse, maximum likelihood estimates can be unreliable. The Bayesian
estimator adds Dirichlet priors, for example pseudo-counts or an equivalent sample size. The
priors smooth the estimated distributions and add prior beliefs.

## Missing Data and Latent Variables

If the dataset has missing values or the model has latent variables, counts of the data are
not sufficient. Then use the Expectation Maximization estimator. In each iteration, it
calculates the expected values of the missing data and then updates the parameters.

## Parallel Estimation

The `n_jobs` argument of estimators such as `DiscreteMLE(n_jobs=...)` estimates the CPDs of
different nodes in parallel. This makes the fit of large models faster. Use the
`state_names` argument to give all states of the variables. You need it if a small dataset
does not contain all states.

## See Also

:::{seealso}
- {doc}`Probabilistic Inference <probabilistic_inference>` — Query the fitted model.
- {doc}`Simulations <simulations>` — Draw synthetic samples from the fitted model.
:::

## API Reference

These pages list all estimators:

- {doc}`Parameter Estimation API <../api/parameter_estimation>`
- {doc}`Models API <../api/models>`
- {doc}`Factors and CPDs API <../api/factors>`
