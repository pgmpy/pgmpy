# Causal Discovery and Structure Learning

```{eval-rst}
.. meta::
   :description: Learn causal graphs from data with the causal discovery algorithms in pgmpy.
```

Causal discovery learns a causal graph from observational data. The graph shows which
variables have a direct causal effect on other variables. pgmpy searches the possible graph
structures and returns a DAG or a PDAG.

## At a Glance

::::{grid} 1 1 2 2
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

All discovery algorithms use the same `fit`, `causal_graph_`, and `score` interface.
:::

:::{grid-item-card} Expert Knowledge
:link: "#expert-knowledge"
:link-type: url
:class-card: sd-card-hover

Use required edges, forbidden edges, and a temporal order to limit the search.
:::

:::{grid-item-card} Scoring and Evaluation
:link: "#scoring-and-evaluation"
:link-type: url
:class-card: sd-card-hover

Use the `score(...)` method to compare a learned graph with the data or with a known graph.
:::

:::{grid-item-card} Scikit-learn Compatibility
:link: "#scikit-learn-compatibility"
:link-type: url
:class-card: sd-card-hover

Use the discovery estimators in scikit-learn pipelines and tools.
:::
::::

## API

All discovery algorithms use the same steps: create the estimator, fit it to the data, and
read the result.

```python
from pgmpy.datasets import load_dataset
from pgmpy.causal_discovery import PC   # swap in any discovery algorithm
from pgmpy.structure_score import BIC

dataset = load_dataset("sachs_discrete")
data = dataset.data

est = PC(ci_test="chi_square", return_type="dag", show_progress=False)
est.fit(data)

print(est.causal_graph_.edges())
print(est.adjacency_matrix_.head())
```

To use a different algorithm, change only the class and its arguments. After the fit,
`causal_graph_` contains the learned graph and `adjacency_matrix_` contains its adjacency
matrix.

## Expert Knowledge

Use the `ExpertKnowledge` class to add domain knowledge to the search. You can specify:

- **Required edges**: edges that the learned graph must contain.
- **Forbidden edges**: edges that the learned graph must not contain.
- **Temporal order**: an order of the variables that limits the edge directions.
- **Search space**: the edges that the algorithm can add to the graph.

Constraint-based and score-based algorithms accept the `expert_knowledge` argument:

```python
from pgmpy.causal_discovery import HillClimbSearch, ExpertKnowledge
from pgmpy.datasets import load_dataset

dataset = load_dataset("sachs_discrete")
data = dataset.data

expert = ExpertKnowledge(
    required_edges=[("PKC", "PKA")],
    forbidden_edges=[("PKA", "PKC")],
    temporal_order=[["PKC", "Raf"], ["Mek", "PKA"]],
)

est = HillClimbSearch(
    scoring_method="bic-d", expert_knowledge=expert, show_progress=False
)
est.fit(data)
print(est.causal_graph_.edges())
```

## Scoring and Evaluation

All discovery estimators have a `score(...)` method. It compares the result with the data or
with a known graph:

```python
from pgmpy.causal_discovery import HillClimbSearch, PC
from pgmpy.datasets import load_dataset
from pgmpy.structure_score import BIC

dataset = load_dataset("sachs_discrete")
data = dataset.data

hc = HillClimbSearch(scoring_method=BIC(data), show_progress=False).fit(data)
pc = PC(ci_test="chi_square", return_type="dag", show_progress=False).fit(data)

# Score against data (no ground truth needed)
print(hc.score(X=data, metric="correlation_score"))
print(pc.score(X=data, metric="correlation_score"))
```

Unsupervised metrics compare the learned graph with the data. Supervised metrics compare it
with a known graph. For `metric`, give a metric name, a metric instance, or `None` for the
default metric. `pgmpy.metrics.get_metric(...)` uses the same rules to find the metric.

## Scikit-learn Compatibility

The discovery estimators are subclasses of `sklearn.base.BaseEstimator`. You can use them
with scikit-learn tools, for example `clone`, `get_params`, and pipelines:

```python
from sklearn.pipeline import Pipeline
from pgmpy.causal_discovery import PC

pipeline = Pipeline(
    steps=[
        ("discover", PC(ci_test="chi_square", return_type="dag", show_progress=False)),
    ]
)
pipeline.fit(data)

print(pipeline[-1].causal_graph_)
```

## Common Recipes

The recipes below use the discrete Sachs dataset:

```python
from pgmpy.datasets import load_dataset

dataset = load_dataset("sachs_discrete")
data = dataset.data
```

### Score-based discovery with BIC

Hill climbing searches the DAGs for the one with the best BIC score. The `bic-d` score is
for discrete data.

```python
from pgmpy.causal_discovery import HillClimbSearch

est = HillClimbSearch(scoring_method="bic-d", show_progress=False).fit(data)
```

### Constraint-based discovery with a different CI test

This recipe uses the G-squared test in PC. For discrete data, PC uses the chi-square test
by default. With `return_type="dag"`, PC also orients the remaining undirected edges.

```python
from pgmpy.causal_discovery import PC

est = PC(ci_test="g_sq", return_type="dag", show_progress=False).fit(data)
```

### Hybrid discovery with MMHC

MMHC finds the skeleton with conditional independence tests. Then it orients the edges
with a score-based search.

```python
from pgmpy.causal_discovery import MMHC

est = MMHC(return_type="dag", show_progress=False).fit(data)
```

### GES returning a PDAG

With `return_type="pdag"`, an edge stays undirected if the data cannot show its direction.

```python
from pgmpy.causal_discovery import GES

est = GES(scoring_method="bic-d", return_type="pdag").fit(data)
```

### Score against the ground truth

Use the structural Hamming distance (SHD) to compare the learned graph with the known graph
of the dataset.

```python
est.score(true_graph=dataset.ground_truth, metric="shd")
```

### Choose the score from the data type

With `scoring_method=None`, the estimator selects the score from the data type. It uses BIC
for discrete data, Gaussian BIC for continuous data, and conditional Gaussian BIC for mixed
data.

```python
continuous_data = load_dataset("sachs_continuous").data

est = HillClimbSearch(scoring_method=None, show_progress=False).fit(continuous_data)
```

## See Also

:::{seealso}
- {doc}`Parameter Estimation <parameter_estimation>` — Estimate the CPDs of a learned graph.
- {doc}`Metrics <metrics>` — Evaluate learned graphs with supervised and unsupervised metrics.
:::

## API Reference

These pages list all discovery algorithms, structure scores, and conditional independence
tests:

- {doc}`Causal Discovery API <../api/structure_learning>`
- {doc}`Conditional Independence Tests API <../api/ci_test>`
- {doc}`Structure Scoring API <../api/structure_score>`
- {doc}`Metrics API <../api/metrics>`
