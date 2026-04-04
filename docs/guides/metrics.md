# Metrics

```{meta}
:description: Evaluate learned graphs and fitted models using pgmpy's metric APIs.
```

After learning a graph or fitting a model, you need to evaluate how good the result is.
pgmpy provides metrics for two settings: **supervised evaluation** compares a learned
graph against a known ground-truth graph, and **unsupervised evaluation** scores a graph
against the observed data when no ground truth is available.

## At a Glance

- **[Unified API](#api)**: All metrics share a consistent callable interface.
- **[Supervised Metrics](#supervised-metrics)**: Compare learned graphs against ground truth using structural distance and confusion matrices.
- **[Unsupervised Metrics](#unsupervised-metrics)**: Score graphs against data using correlation residuals, implied conditional independencies, and structure scores.
- **[Metric Discovery](#metric-discovery)**: Programmatically find available metrics by their requirements.

## API

All metrics follow the same pattern — instantiate and call:

```python
from pgmpy.base import DAG
from pgmpy.metrics import SHD

true_graph = DAG([("A", "B"), ("B", "C")])
estimated_graph = DAG([("B", "A"), ("B", "C")])

metric = SHD()
print(metric(true_causal_graph=true_graph, est_causal_graph=estimated_graph))
```

## Supervised Metrics

When a ground-truth graph is available, supervised metrics measure the structural
distance between the estimated and true graphs — counting edge additions, deletions,
and reversals, or computing confusion matrices for adjacency and orientation accuracy.

## Unsupervised Metrics

When no ground truth is available, unsupervised metrics score how well a graph explains
the observed data. These check whether the conditional independencies implied by the
graph hold in the data, or evaluate the graph under a configurable structure scoring
method.

## Metric Discovery

`get_metrics(...)` lets you discover available metrics by their requirements instead of
hard-coding class names:

```python
from pgmpy.metrics import get_metrics

supervised = get_metrics(requires_true_graph=True)
unsupervised = get_metrics(requires_data=True)
```

## See Also

:::{seealso}
- {doc}`Causal Discovery <causal_discovery>` — The primary workflow that produces graphs for evaluation.
:::

## API Reference

For the full list of supported metrics:

- {doc}`Metrics API <../api/metrics>`
- {doc}`Graph Classes API <../api/base>`
