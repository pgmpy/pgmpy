# Extensibility

```{meta}
:description: Extend pgmpy using the repository templates for new datasets, models, metrics, and discovery algorithms.
```

pgmpy is designed to be extensible. When you want to add a new algorithm, dataset,
metric, or model, repository templates ensure that new contributions integrate seamlessly
with the existing API — discoverable through the same listing and filtering functions.

## At a Glance

- **[Unified API](#api)**: A template-driven workflow that is the same across all extension types.
- **[Extension Templates](#extension-templates)**: Ready-made scaffolds for discovery algorithms, datasets, example models, and metrics.
- **[Automatic Registration](#automatic-registration)**: New extensions become part of the public API once exported from the package.

## API

The extension workflow follows a consistent pattern:

1. Pick a template from `devtools/extension_templates/`.
2. Copy it into the correct package directory.
3. Fill in the TODO markers and implementation details.
4. Export the new object from the package `__init__.py`.
5. Add the matching tests and docs.

```bash
ls devtools/extension_templates
cp devtools/extension_templates/_metrics.py pgmpy/metrics/my_metric.py
```

## Extension Templates

Templates are available for the most common extension types:

- **Discovery algorithms**: New estimators follow the unified `fit` / `causal_graph_` /
  `score` API.
- **Datasets**: New datasets become discoverable via `list_datasets()`.
- **Example models**: New models become discoverable via `list_models()`.
- **Metrics**: New metrics become discoverable via `get_metrics()`.

## Automatic Registration

Each extension needs an entry in its package's `__init__.py` to become part of the public
API. Once registered, the new object is automatically discoverable through the listing
and filtering functions used by the rest of the library.

## API Reference

For existing components that serve as reference implementations:

- {doc}`Causal Discovery API <../api/structure_learning>`
- {doc}`Datasets and Example Models API <../api/data>`
- {doc}`Metrics API <../api/metrics>`
- {doc}`Models API <../api/models>`
