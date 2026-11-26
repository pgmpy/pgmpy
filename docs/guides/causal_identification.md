# Causal Identification

```{eval-rst}
.. meta::
   :description: Find adjustment sets and frontdoor sets in causal graphs with pgmpy.
```

Before you estimate a causal effect from data, make sure that the effect is identifiable.
Causal identification answers this question: is it possible to calculate the causal effect
of X on Y from the distribution of the observed data only? The answer depends on the graph
structure. Unobserved confounders can make an effect not identifiable.

## At a Glance

::::{grid} 1 1 2 2
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

All identification methods use the same `identify(graph)` and `validate(graph)` methods.
:::

:::{grid-item-card} Backdoor Adjustment
:link: "#backdoor-adjustment"
:link-type: url
:class-card: sd-card-hover

Find minimal, variance-optimal, or all valid adjustment sets.
:::

:::{grid-item-card} Frontdoor Identification
:link: "#frontdoor-identification"
:link-type: url
:class-card: sd-card-hover

Identify effects when unobserved confounders prevent backdoor adjustment.
:::

:::{grid-item-card} Validation
:link: "#validation"
:link-type: url
:class-card: sd-card-hover

Check if a candidate adjustment set or frontdoor set is valid.
:::
::::

## API

All identification methods use the same steps: create the identifier, call
`identify(...)`, and examine the result:

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

The backdoor criterion finds sets of variables that block all non-causal paths between the
exposure and the outcome. You condition on these variables. pgmpy can find a minimal
adjustment set, a variance-optimal set, or all valid sets:

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

If unobserved confounders prevent backdoor adjustment, use the frontdoor criterion. It
identifies the effect through mediator variables that carry the effect from the exposure to
the outcome.

## Validation

If you have a candidate adjustment set or frontdoor set, use `validate(...)`. It checks if
the set satisfies the graphical criterion. Do this before the estimation:

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

These pages list all identification methods:

- {doc}`Causal Inference API <../api/causal_inference>`
- {doc}`Graph Classes API <../api/base>`
