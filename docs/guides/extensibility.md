# Extensibility

```{eval-rst}
.. meta::
   :description: Use the repository templates to add datasets, models, metrics, and discovery algorithms to pgmpy.
```

You can add new algorithms, datasets, metrics, and models to pgmpy. Start from a template in
the repository. The template makes sure that the new object uses the same API as the
existing objects.

## At a Glance

::::{grid} 1 1 3 3
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

The steps are the same for all types of extension.
:::

:::{grid-item-card} Extension Templates
:link: "#extension-templates"
:link-type: url
:class-card: sd-card-hover

Templates for discovery algorithms, CI tests, structure scores, datasets, example models, and
metrics.
:::

:::{grid-item-card} Automatic Registration
:link: "#automatic-registration"
:link-type: url
:class-card: sd-card-hover

pgmpy finds new datasets, example models, metrics, and structure scores from their files.
:::
::::

## API

To add an extension, do these steps:

1. Select a template from `devtools/extension_templates/`.
2. Copy it to the correct package directory.
3. Complete the TODO items and write the implementation.
4. If the template tells you to, import the new object in the `__init__.py` of the package.
5. Add tests and documentation for the new object.

```bash
ls devtools/extension_templates
cp devtools/extension_templates/_metrics.py pgmpy/metrics/my_metric.py
```

## Extension Templates

The repository has templates for these types of extension:

- **Discovery algorithms**: New estimators use the same `fit`, `causal_graph_`, and `score`
  API.
- **CI tests** and **structure scores**: New tests and scores become available to the
  discovery algorithms.
- **Datasets**: `list_datasets()` finds new datasets.
- **Example models**: `list_models()` finds new models.
- **Metrics**: `get_metric()` can select new metrics by name.

## Automatic Registration

pgmpy finds datasets, example models, metrics, and structure scores from their files. Do not
start the file name with an underscore, because pgmpy does not find these files. For
discovery algorithms, CI tests, and metrics, also add an import to the `__init__.py` of the
package. Each template gives the exact steps.

## API Reference

Use the existing objects on these pages as reference implementations:

- {doc}`Causal Discovery API <../api/structure_learning>`
- {doc}`Datasets and Example Models API <../api/data>`
- {doc}`Metrics API <../api/metrics>`
- {doc}`Models API <../api/models>`
