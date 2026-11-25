from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def _format_tag_type(tag_type: str | tuple) -> str:
    if isinstance(tag_type, str):
        return f"``{tag_type}``"
    # Class-valued choices are graph classes, which are documented under pgmpy.base.
    choices = ", ".join(
        f":class:`~pgmpy.base.{choice.__name__}`" if isinstance(choice, type) else f"``{choice!r}``"
        for choice in tag_type[1]
    )
    return f"one of {choices}" if tag_type[0] == "str" else f"list with elements from {choices}"


def render_tag_reference(object_type: str) -> list[str]:
    """Render the tags of ``object_type`` from :mod:`pgmpy.registry` as an RST definition list.

    Each entry shows the tag name, its value type, and the docstring of its tag class. Used by the
    ``.. pgmpy-tags:: <object_type>`` directive.
    """
    from pgmpy.registry import _tags as tag_module

    lines: list[str] = []
    for tag_name, _, tag_type, _ in tag_module.all_tags(object_type):
        doc = inspect.cleandoc(getattr(tag_module, tag_name.replace(":", "__")).__doc__)
        lines += [f"``{tag_name}``", f"   **Type:** {_format_tag_type(tag_type)}", ""]
        lines += [f"   {line}" if line else "" for line in doc.splitlines()]
        lines.append("")
    return lines


def _rst_list_table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Render ``header`` and ``rows`` as the lines of an RST ``list-table`` directive."""
    lines = [".. list-table::", "   :header-rows: 1", ""]
    for row in [header, *rows]:
        lines.append(f"   * - {row[0]}")
        lines += [f"     - {cell}".rstrip() for cell in row[1:]]
    lines.append("")
    return lines


def _sorted_class_tags(base_class: type, package_name: str) -> list[dict[str, Any]]:
    """Return the tags of every named ``base_class`` subclass in ``package_name``, sorted by name."""
    from skbase.lookup import all_objects

    classes = all_objects(object_types=base_class, package_name=package_name, return_names=False)
    return sorted(
        (cls.get_class_tags() for cls in classes if cls.get_class_tag("name") is not None),
        key=lambda tags: tags["name"],
    )


def render_dataset_catalog() -> list[str]:
    """Render every dataset listed by :func:`pgmpy.datasets.list_datasets` as an RST table of its tags."""
    from pgmpy.datasets._base import BaseDataset

    type_labels = [
        ("is_discrete", "Discrete"),
        ("is_continuous", "Continuous"),
        ("is_mixed", "Mixed"),
        ("is_ordinal", "Ordinal"),
    ]
    flag_columns = [
        ("has_ground_truth", "Ground truth"),
        ("has_expert_knowledge", "Expert knowledge"),
        ("has_missing_data", "Missing data"),
        ("is_simulated", "Simulated"),
        ("is_interventional", "Interventional"),
    ]
    rows = [
        [
            f"``{tags['name']}``",
            "varies" if tags["n_variables"] is None else f"{tags['n_variables']:,}",
            "varies" if tags["n_samples"] is None else f"{tags['n_samples']:,}",
            ", ".join(label for tag, label in type_labels if tags[tag]),
            *("✓" if tags[tag] else "" for tag, _ in flag_columns),
        ]
        for tags in _sorted_class_tags(BaseDataset, "pgmpy.datasets")
    ]
    header = ["Name", "Variables", "Samples", "Type", *(label for _, label in flag_columns)]
    return _rst_list_table(header, rows)


def render_example_model_catalog() -> list[str]:
    """Render every model listed by :func:`pgmpy.example_models.list_models` as an RST table of its tags."""
    from pgmpy.example_models._base import BaseExampleModel

    rows = []
    for tags in _sorted_class_tags(BaseExampleModel, "pgmpy.example_models"):
        if not tags["is_parameterized"]:
            model_type = "Structure only"
        elif tags["is_hybrid"]:
            model_type = "Hybrid"
        elif tags["is_discrete"]:
            model_type = "Discrete"
        else:
            model_type = "Continuous"
        rows.append([f"``{tags['name']}``", f"{tags['n_nodes']:,}", f"{tags['n_edges']:,}", model_type])
    return _rst_list_table(["Name", "Nodes", "Edges", "Type"], rows)


def render_catalog(kind: str) -> list[str]:
    """Render the table of all available ``"datasets"`` or ``"example_models"``.

    Used by the ``.. pgmpy-catalog:: <kind>`` directive.
    """
    renderers = {"datasets": render_dataset_catalog, "example_models": render_example_model_catalog}
    return renderers[kind]()


def rst_lines_directive(render: Callable[[str], list[str]]) -> type:
    """Build a directive that parses the RST lines ``render`` returns for the directive's single argument."""
    from docutils import nodes
    from docutils.statemachine import StringList
    from sphinx.util.docutils import SphinxDirective
    from sphinx.util.nodes import nested_parse_with_titles

    class RstLinesDirective(SphinxDirective):
        required_arguments = 1

        def run(self) -> list[Any]:
            container = nodes.container()
            content = StringList(render(self.arguments[0]), source=f"{render.__name__}({self.arguments[0]})")
            nested_parse_with_titles(self.state, content, container)
            return container.children

    return RstLinesDirective
