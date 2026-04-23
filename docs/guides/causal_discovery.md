# Causal Discovery and Structure Learning

```{meta}
:description: Learn causal graphs from data using the unified causal discovery APIs in pgmpy.
```

Causal discovery recovers the causal graph from observational data — determining which
variables directly cause which others. Given a dataset of jointly observed variables,
pgmpy searches over graph structures to produce a DAG or PDAG representing the causal
relationships.

## At a Glance

- **[Unified API](#api)**: All discovery algorithms share a consistent `fit` / `causal_graph_` / `score` interface.
- **[Expert Knowledge](#expert-knowledge)**: Encode required edges, forbidden edges, and temporal orderings to constrain the search.
- **[Scoring and Evaluation](#scoring-and-evaluation)**: Compare learned graphs against data or a known reference using a unified `score(...)` method.
- **[Scikit-learn Compatibility](#scikit-learn-compatibility)**: Use discovery estimators in sklearn pipelines and tooling.

## API

All discovery algorithms follow the same pattern — instantiate, fit, and read the result:

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

Switching algorithms requires only changing the class and its constructor arguments.
The fitted result is always accessed through `causal_graph_` and `adjacency_matrix_`.

## Expert Knowledge

pgmpy supports incorporating domain knowledge into the discovery process through the
`ExpertKnowledge` class. You can encode:

- **Required edges** that must appear in the learned graph.
- **Forbidden edges** that must not appear.
- **Temporal orderings** that constrain edge directions.
- **Search-space restrictions** that limit which variables are considered.

Both constraint-based and score-based algorithms accept an `expert_knowledge` parameter:

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

All discovery estimators expose a unified `score(...)` method for comparing results:

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

You can score against data using unsupervised metrics, or against a known reference
graph using supervised metrics. Use `pgmpy.metrics.get_metrics(...)` to discover
available metrics programmatically.

## Scikit-learn Compatibility

Discovery estimators inherit from `sklearn.base.BaseEstimator` and work with sklearn
tooling — cloning, parameter inspection, and pipelines:

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

```python
from pgmpy.datasets import load_dataset

dataset = load_dataset("sachs_discrete")
data = dataset.data
```

**Score-based discovery with BIC:**
```python
from pgmpy.causal_discovery import HillClimbSearch

est = HillClimbSearch(scoring_method="bic-d", show_progress=False).fit(data)
```

**Constraint-based discovery with a different CI test:**
```python
from pgmpy.causal_discovery import PC

est = PC(ci_test="g_sq", return_type="dag", show_progress=False).fit(data)
```

**GES returning a PDAG:**
```python
from pgmpy.causal_discovery import GES

est = GES(scoring_method="bic-d", return_type="pdag", show_progress=False).fit(data)
```

**Score against ground truth:**
```python
est.score(true_graph=dataset.ground_truth, metric="shd")
```

**Auto-detect scoring for continuous data:**
```python
est = HillClimbSearch(scoring_method=None, show_progress=False).fit(continuous_data)
```

## See Also

:::{seealso}
- {doc}`Parameter Estimation <parameter_estimation>` — Estimate CPDs once you have a graph.
- {doc}`Metrics <metrics>` — Evaluate learned graphs with supervised and unsupervised metrics.
:::

## API Reference

For the full list of discovery algorithms, scoring methods, and conditional independence
tests:

- {doc}`Causal Discovery API <../api/structure_learning>`
- {doc}`Conditional Independence Tests API <../api/ci_test>`
- {doc}`Structure Scoring API <../api/structure_score>`
- {doc}`Metrics API <../api/metrics>`
