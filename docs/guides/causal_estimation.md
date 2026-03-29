# Causal Estimation

```{meta}
:description: Estimate causal effects from graphs and data using pgmpy's causal inference and prediction APIs.
```

Causal estimation turns an identified causal query into a numerical effect. Once you know
that a causal effect is identifiable (see {doc}`Causal Identification <causal_identification>`),
this step combines the graph and observed data to compute quantities like the average
treatment effect or an interventional distribution.

:::{note}
**Prerequisite:** This guide assumes a causal effect has been identified. See
{doc}`Causal Identification <causal_identification>` to determine identifiability first.
:::

:::{tip}
**When to use this vs. Probabilistic Inference:** Use *Causal Estimation* when you need
interventional answers (do-calculus, "what if I set X to x?"). Use
{doc}`Probabilistic Inference <probabilistic_inference>` for observational queries
(conditioning on evidence, "what is P(Y | X=x)?").
:::

## At a Glance

- **[Unified API](#api)**: Two entry points — `CausalInference` for graph-based queries, and `pgmpy.prediction` for sklearn-style regression.
- **[Interventional Queries](#interventional-queries)**: Compute distributions under do-interventions on fitted discrete models.
- **[Average Treatment Effect](#average-treatment-effect)**: Estimate ATE from data using identified adjustment sets.
- **[Regression-Based Estimators](#regression-based-estimators)**: Sklearn-compatible regressors for adjustment, instrumental variable, and double ML workflows.

## API

pgmpy provides two consistent entry points:

**Graph-based queries** using `CausalInference`:

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

**Sklearn-style regressors** using `pgmpy.prediction` estimators that use causal graph
role annotations (exposures, outcomes, adjustment sets) to drive the estimation.

## Interventional Queries

For fitted discrete models, `CausalInference.query(..., do=..., evidence=...)` computes
interventional distributions such as P(Y | do(X), Z), letting you answer "what if"
questions directly from the model:

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
on an outcome, automatically using identified adjustment sets from the causal graph.

## Regression-Based Estimators

pgmpy provides sklearn-compatible regressors that use causal graph roles to determine
which variables play which role in the estimation. These include adjustment-based
regression, instrumental variable regression, and cross-fitted double machine learning
for flexible nuisance models:

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
- {doc}`Causal Identification <causal_identification>` — Verify identifiability before estimation.
:::

## API Reference

For the full list of estimation methods:

- {doc}`Causal Inference API <../api/causal_inference>`
- {doc}`Graph Classes API <../api/base>`
