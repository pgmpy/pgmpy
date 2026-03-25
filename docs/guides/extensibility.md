# Extensibility

```{meta}
:description: Extend pgmpy with new datasets, example models, metrics, and causal discovery algorithms using the repository templates.
```

pgmpy includes repository templates for adding new extension points without
starting from a blank file.

These templates live in `devtools/extension_templates/` and are the right
starting point when you want to add a new dataset, example model, metric, or
causal discovery algorithm.

## When to use

- Use these templates when you want to add a new public extension to pgmpy in a
  way that matches the existing package structure.
- Use them before writing a new module from scratch, because they already
  encode the expected imports, class shape, metadata tags, and contribution
  checklist.
- Use them together with the test layout under `pgmpy/tests/` when preparing a
  contribution.

## Available Templates

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Extension Type
     - Template
     - Target Location
   * - Causal discovery algorithm
     - ``devtools/extension_templates/_causal_discovery.py``
     - ``pgmpy/causal_discovery/``
   * - Dataset
     - ``devtools/extension_templates/_dataset.py``
     - ``pgmpy/datasets/``
   * - Example model
     - ``devtools/extension_templates/_example_model.py``
     - ``pgmpy/example_models/``
   * - Metric
     - ``devtools/extension_templates/_metrics.py``
     - ``pgmpy/metrics/``
```

## Typical Workflow

1. Pick the closest template from `devtools/extension_templates/`.
2. Copy it into the target package and rename it to the final public module
   name.
3. Work through the `TODO` markers in the template.
4. Register the new object in the package `__init__.py` if the template asks
   for it.
5. Add tests in the corresponding `pgmpy/tests/` area before contributing.

One important detail is common across these templates: the template filenames
start with `_` because they are not meant to be imported directly. Your real
extension module should use a normal public filename.

## What Each Template Covers

### Adding a Causal Discovery Algorithm

The causal discovery template defines a `MyCausalDiscoveryAlgo` skeleton based
on `_BaseCausalDiscovery`, with placeholders for hyperparameters, learned graph
attributes, and the `_fit` implementation.

Use `devtools/extension_templates/_causal_discovery.py` when you want to add a
new discovery estimator under `pgmpy/causal_discovery/`. The template also
calls out the matching test location under `pgmpy/tests/test_causal_discovery/`.

### Adding a Dataset

The dataset template defines a dataset class based on `_BaseDataset` and walks
through the required `_tags`, remote asset URLs, parsing hooks, and optional
expert knowledge / ground-truth loaders.

Use `devtools/extension_templates/_dataset.py` when you want the dataset to be
discoverable through `list_datasets()` and loadable through `load_dataset()`.

### Adding an Example Model

The example-model template explains how to choose the right mixin, define model
metadata, and point to the backing asset for a parameterized model or DAG.

Use `devtools/extension_templates/_example_model.py` when adding a model under
`pgmpy/example_models/`, including a new source subdirectory if needed.

### Adding a Metric

The metrics template includes both supervised and unsupervised skeletons and
documents the required tags, base classes, and evaluation method shape.

Use `devtools/extension_templates/_metrics.py` when adding a new graph metric
under `pgmpy/metrics/`. It also documents the expected exports and test file
layout.

## Example

For a new metric implementation, the high-level flow is:

```text
1. Copy devtools/extension_templates/_metrics.py to pgmpy/metrics/my_metric.py
2. Remove the unused template class
3. Fill in _tags, __init__, and _evaluate
4. Export the metric from pgmpy/metrics/__init__.py
5. Add tests in pgmpy/tests/test_metrics/test_my_metric.py
```

## See Also

- **Related guides:** {doc}`custom_model` | {doc}`datasets` | {doc}`example_models` | {doc}`metrics` | {doc}`causal_discovery`
- **Contributing:** {doc}`../development`
- **Previous:** {doc}`plotting` -- visualize graphs with pygraphviz, daft, and networkx
