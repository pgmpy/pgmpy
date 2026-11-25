"""Sphinx extension for the pgmpy documentation site.

- ``versions``: site configuration and the docs versions manifest.
- ``staging``: copying notebook sources into ``docs/`` before a build.
- ``seo``: page metadata, structured data and ``llms.txt``.
- ``docstrings``: autodoc docstring processing.
- ``directives``: directives that render content generated from pgmpy.
"""

from __future__ import annotations

from typing import Any

from .directives import render_catalog, render_tag_reference, rst_lines_directive
from .docstrings import expand_references, skip_sklearn_metadata_request_method
from .seo import on_build_finished, on_builder_inited, on_html_page_context
from .staging import stage_docs_sources
from .versions import (
    build_versions_payload,
    current_version_label,
    load_versions_manifest,
    resolve_release_docs_metadata,
    resolve_site_config,
)

__all__ = [
    "build_versions_payload",
    "current_version_label",
    "load_versions_manifest",
    "resolve_release_docs_metadata",
    "resolve_site_config",
    "setup",
    "stage_docs_sources",
]


def setup(app: Any) -> dict[str, Any]:
    app.add_directive("pgmpy-tags", rst_lines_directive(render_tag_reference))
    app.add_directive("pgmpy-catalog", rst_lines_directive(render_catalog))
    app.connect("builder-inited", on_builder_inited)
    app.connect("html-page-context", on_html_page_context)
    app.connect("build-finished", on_build_finished)
    # priority < numpydoc's default (500) so we edit the raw numpydoc docstring
    app.connect("autodoc-process-docstring", expand_references, priority=400)
    app.connect("autodoc-skip-member", skip_sklearn_metadata_request_method)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
