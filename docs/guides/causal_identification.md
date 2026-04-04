# Causal Identification

```{meta}
:description: Identify adjustment and frontdoor strategies from causal graphs using pgmpy's identification APIs.
```

Before estimating a causal effect from data, you need to determine whether the effect
is even identifiable from observational data given the causal graph. Causal identification
answers: "Can the causal effect of X on Y be expressed purely in terms of the observed
data distribution?" The answer depends on the graph structure — if there are unobserved
confounders, certain effects may not be identifiable.

## At a Glance

- **[Unified API](#api)**: All identification methods share an `identify(graph)` / `validate(graph)` interface.
- **[Backdoor Adjustment](#backdoor-adjustment)**: Find minimal, variance-optimal, or all valid adjustment sets.
- **[Frontdoor Identification](#frontdoor-identification)**: Identify effects when backdoor adjustment is blocked by unobserved confounders.
- **[Validation](#validation)**: Verify whether a candidate adjustment or frontdoor set is valid.

## API

Identification methods follow a consistent pattern — create the identifier, call
`identify(...)`, and inspect the result:

```python
from pgmpy.base import DAG
from pgmpy.identification import Adjustment  # swap in any identification method

dag = DAG(
    [("X", "Y"), ("Z", "X"), ("Z", "Y")],
    roles={"exposures": "X", "outcomes": "Y"},
)

identified_graph, success = Adjustment(variant="minimal").identify(dag)

print(success)
print(identified_graph.get_role("adjustment"))
```

## Backdoor Adjustment

The backdoor criterion finds sets of variables to condition on that block all spurious
paths between exposure and outcome. pgmpy supports finding minimal adjustment sets,
variance-optimal sets, or enumerating all valid sets:

```python
from pgmpy.base import DAG
from pgmpy.identification import Adjustment

dag = DAG(
    [("X", "Y"), ("Z", "X"), ("Z", "Y")],
    roles={"exposures": "X", "outcomes": "Y"},
)

# Minimal adjustment set
identified_min, ok_min = Adjustment(variant="minimal").identify(dag)
print("Minimal:", identified_min.get_role("adjustment"), ok_min)

# All valid adjustment sets
identified_all, ok_all = Adjustment(variant="all").identify(dag)
print("All:", identified_all.get_role("adjustment"), ok_all)
```

## Frontdoor Identification

When unobserved confounders block the backdoor criterion, the frontdoor criterion provides
an alternative by identifying mediating variables through which the causal effect flows.

## Validation

If you already have a candidate adjustment or frontdoor set, use `validate(...)` to check
whether it satisfies the graphical criterion before proceeding to estimation:

```python
from pgmpy.base import DAG
from pgmpy.identification import Adjustment

dag = DAG(
    [("X", "Y"), ("Z", "X"), ("Z", "Y")],
    roles={"exposures": "X", "outcomes": "Y", "adjustment": "Z"},
)

is_valid = Adjustment(variant="minimal").validate(dag)
print(is_valid)  # True
```

## See Also

:::{seealso}
- {doc}`Causal Estimation <causal_estimation>` — Estimate the identified causal effects from data.
:::

## API Reference

For the full list of identification methods:

- {doc}`Causal Inference API <../api/causal_inference>`
- {doc}`Graph Classes API <../api/base>`
