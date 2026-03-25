# Causal Estimation

```{meta}
:description: Estimate causal effects from data after identification using do-calculus and regression-based estimators.
```

Causal estimation quantifies how much changing a variable changes an outcome.

More precisely, it estimates causal effects such as the average treatment
effect (ATE) from observed data using an identified adjustment set:

```{math}
ATE = E[Y \mid do(X=1)] - E[Y \mid do(X=0)]
```

## When to use

- Use `CausalInference` when you already have a causal graph and want
  interventional quantities from that graph.
- Use the naive adjustment regressor as a simple baseline for backdoor-based
  estimation.
- Use the naive IV regressor when an instrumental variable assumption is more
  plausible than unconfoundedness.
- Use Double ML when you want a more flexible semi-parametric estimator for
  high-dimensional settings.

## Example

```python
from pgmpy.datasets import load_dataset
from pgmpy.inference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork

dataset = load_dataset("sachs_discrete")
data = dataset.data
dag = DiscreteBayesianNetwork(
    [
        ("PKA", "ERK"),
        ("ERK", "Akt"),
        ("PKA", "Akt"),
    ]
)
ci = CausalInference(dag)
ate = ci.estimate_ate("PKA", "Akt", data)
print(ate)
```

## Algorithms

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Algorithm
     - API Reference
   * - CausalInference (do-calculus)
     - :class:`pgmpy.inference.CausalInference.CausalInference`
```

## Semi-parametric Estimators

These methods combine graphical structure with flexible regression models for
heterogeneous treatment effect estimation.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Algorithm
     - API Reference
   * - Naive Adjustment Regressor
     - :class:`pgmpy.prediction.NaiveAdjustmentRegressor.NaiveAdjustmentRegressor`
   * - Naive IV Regressor
     - :class:`pgmpy.prediction.NaiveIVRegressor.NaiveIVRegressor`
   * - Double ML Regressor
     - :class:`pgmpy.prediction.DoubleMLRegressor.DoubleMLRegressor`
```

## See Also

- **Examples:** {doc}`Causal Inference <../examples/Causal_Inference>` | {doc}`Causal Games <../examples/Causal_Games>`
- **API Reference:** {doc}`Causal Inference API <../api/causal_inference>`
- **Previous:** {doc}`causal_identification` -- check identifiability
- **Next:** {doc}`metrics` -- evaluate the learned model
