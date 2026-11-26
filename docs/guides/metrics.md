# Metrics

```{eval-rst}
.. meta::
   :description: Evaluate learned graphs and fitted models with the metrics in pgmpy.
```

After you learn a graph or fit a model, evaluate the result. pgmpy has metrics for two
cases:

- **Supervised evaluation** compares a learned graph with a known ground-truth graph.
- **Unsupervised evaluation** compares a graph with the observed data. Use it when you do not
  have a ground truth.

## At a Glance

::::{grid} 1 1 2 2
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

You call all metrics in the same way.
:::

:::{grid-item-card} Supervised Metrics
:link: "#supervised-metrics"
:link-type: url
:class-card: sd-card-hover

Compare a learned graph with the ground truth. Use structural distances and confusion
matrices.
:::

:::{grid-item-card} Unsupervised Metrics
:link: "#unsupervised-metrics"
:link-type: url
:class-card: sd-card-hover

Compare a graph with the data. Use correlation residuals, implied conditional
independencies, and structure scores.
:::

:::{grid-item-card} Metric Discovery
:link: "#metric-discovery"
:link-type: url
:class-card: sd-card-hover

Find the available metrics in code by their requirements.
:::
::::

## API

All metrics use the same steps: create the metric, then call it:

```python
from pgmpy.base import DAG
from pgmpy.metrics import SHD

true_graph = DAG([("A", "B"), ("B", "C")])
estimated_graph = DAG([("B", "A"), ("B", "C")])

metric = SHD()
print(metric(true_causal_graph=true_graph, est_causal_graph=estimated_graph))
```

## Supervised Metrics

If you have a ground-truth graph, use a supervised metric. Some supervised metrics measure
the structural distance between the estimated graph and the true graph. They count the
added, removed, and reversed edges. Other supervised metrics calculate confusion matrices
for the adjacencies and the edge orientations.

## Unsupervised Metrics

If you do not have a ground truth, use an unsupervised metric. It measures how well a graph
explains the observed data. Some unsupervised metrics check if the conditional
independencies of the graph are true in the data. Other metrics evaluate the graph with a
structure score that you select.

## Metric Discovery

`get_metric(...)` returns a metric by name, or the default metric for a type of comparison:

```python
from pgmpy.metrics import get_metric

shd = get_metric("shd")
default_unsupervised = get_metric(requires_true_graph=False)
```

To list metrics by their tags, use `skbase.lookup.all_objects`:

```python
from skbase.lookup import all_objects
from pgmpy.metrics import BaseSupervisedMetric, BaseUnsupervisedMetric

supervised = all_objects(object_types=BaseSupervisedMetric, package_name="pgmpy.metrics", return_names=False)
unsupervised = all_objects(object_types=BaseUnsupervisedMetric, package_name="pgmpy.metrics", return_names=False)
```

## See Also

:::{seealso}
- {doc}`Causal Discovery <causal_discovery>` — Learn the graphs that you evaluate.
:::

## API Reference

These pages list all metrics:

- {doc}`Metrics API <../api/metrics>`
- {doc}`Graph Classes API <../api/base>`
