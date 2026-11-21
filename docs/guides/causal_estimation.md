# Causal Estimation

```{eval-rst}
.. meta::
   :description: Estimate causal effects from a graph and data with pgmpy.
```

Causal estimation calculates the numerical value of an identified causal effect. It uses
the graph and the observed data. Examples are the average treatment effect and an
interventional distribution.

:::{note}
**Prerequisite:** The causal effect must be identifiable. Use
{doc}`Causal Identification <causal_identification>` to check this first.
:::

:::{tip}
**When to use this vs. Probabilistic Inference:** Use *Causal Estimation* for
interventional questions (do-calculus, "what if I set X to x?"). Use
{doc}`Probabilistic Inference <probabilistic_inference>` for observational questions that
condition on evidence ("what is P(Y | X=x)?").
:::

## At a Glance

::::{grid} 1 1 2 2
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

Use `CausalInference` for graph-based queries and `pgmpy.prediction` for scikit-learn
regression.
:::

:::{grid-item-card} Interventional Queries
:link: "#interventional-queries"
:link-type: url
:class-card: sd-card-hover

Calculate distributions under do-interventions for fitted discrete models.
:::

:::{grid-item-card} Average Treatment Effect
:link: "#average-treatment-effect"
:link-type: url
:class-card: sd-card-hover

Estimate the average treatment effect from data with an identified adjustment set.
:::

:::{grid-item-card} Regression-Based Estimators
:link: "#regression-based-estimators"
:link-type: url
:class-card: sd-card-hover

Scikit-learn regressors for adjustment, instrumental variables, and double machine learning.
:::
::::

## API

pgmpy has two entry points:

**Graph-based queries** with `CausalInference`:

```python
import numpy as np
import pandas as pd
from pgmpy.base import DAG
from pgmpy.inference import CausalInference

rng = np.random.default_rng(42)
data = pd.DataFrame(
    {
        "X": rng.normal(size=500),
        "Z": rng.normal(size=500),
        "Y": rng.normal(size=500),
    }
)

graph = DAG([("X", "Y"), ("Z", "X"), ("Z", "Y")])
ci = CausalInference(graph)

ate = ci.estimate_ate("X", "Y", data=data, estimator_type="linear")
print(round(float(ate), 4))
```

**Scikit-learn regressors** in `pgmpy.prediction`. These estimators use the roles in the
causal graph (exposures, outcomes, adjustment sets) for the estimation.

## Interventional Queries

For fitted discrete models, `CausalInference.query(..., do=..., evidence=...)` calculates
interventional distributions such as P(Y | do(X), Z). Use it to answer "what if" questions
from the model:

```python
from pgmpy.example_models import load_model
from pgmpy.inference import CausalInference

model = load_model("bnlearn/alarm")
ci = CausalInference(model)

result = ci.query(
    variables=["HISTORY"],
    do={"CVP": "LOW"},
    evidence={"PCWP": "LOW"},
    show_progress=False,
)
print(result)
```

## Average Treatment Effect

`CausalInference.estimate_ate(...)` estimates the average treatment effect of an exposure
on an outcome. It finds an adjustment set in the causal graph automatically.

## Regression-Based Estimators

The regressors in `pgmpy.prediction` are compatible with scikit-learn. They use the roles in
the causal graph to find the exposure, the outcome, and the other variables. The regressors
include:

- adjustment-based regression.
- instrumental variable regression.
- cross-fitted double machine learning.

This example uses adjustment-based regression:

```python
import numpy as np
import pandas as pd
from pgmpy.base import DAG
from pgmpy.prediction import NaiveAdjustmentRegressor

rng = np.random.default_rng(42)
Z = rng.normal(size=500)
X = 0.5 * Z + rng.normal(size=500)
Y = 0.8 * X + 0.3 * Z + rng.normal(size=500)
data = pd.DataFrame({"X": X, "Z": Z, "Y": Y})

graph = DAG(
    [("X", "Y"), ("Z", "X"), ("Z", "Y")],
    roles={"exposures": "X", "outcomes": "Y", "adjustment": "Z"},
)

reg = NaiveAdjustmentRegressor(causal_graph=graph)
reg.fit(data[["X", "Z"]], data["Y"])
print(reg.predict(data[["X", "Z"]])[:5])
```

## See Also

:::{seealso}
- {doc}`Causal Identification <causal_identification>` — Make sure that the effect is identifiable before the estimation.
:::

## API Reference

These pages list all estimation methods:

- {doc}`Causal Inference API <../api/causal_inference>`
- {doc}`Graph Classes API <../api/base>`
