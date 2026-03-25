import importlib.util
import sys
from pathlib import Path


def _load_docs_module():
    module_path = Path(__file__).parent.parent.parent.parent / "docs" / "_ext" / "pgmpy_docs.py"
    spec = importlib.util.spec_from_file_location("pgmpy_docs", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_site_config_environments():
    docs_module = _load_docs_module()

    production = docs_module.resolve_site_config({"PGMPY_DOCS_ENV": "production"})
    assert production.base_url == "https://pgmpy.org"
    assert production.is_indexable is True
    assert production.robots_meta == "index,follow,max-image-preview:large"

    preview = docs_module.resolve_site_config(
        {
            "PGMPY_DOCS_ENV": "preview",
            "PGMPY_DOCS_BASEURL": "https://preview.pgmpy.org",
        }
    )
    assert preview.base_url == "https://preview.pgmpy.org"
    assert preview.is_indexable is False
    assert preview.robots_meta == "noindex,nofollow,noarchive"


def test_discover_pages_extracts_primary_sections_and_descriptions():
    docs_module = _load_docs_module()
    docs_root = Path(__file__).parent.parent.parent.parent / "docs"

    pages = docs_module.discover_pages(docs_root)
    by_docname = {page.docname: page for page in pages}

    for docname in {"index", "documentation", "reference", "examples", "started/index"}:
        assert docname in by_docname
        assert by_docname[docname].description


def test_render_llms_document_and_structured_data():
    docs_module = _load_docs_module()
    docs_root = Path(__file__).parent.parent.parent.parent / "docs"
    site_config = docs_module.resolve_site_config({"PGMPY_DOCS_ENV": "production"})
    pages = docs_module.discover_pages(docs_root)
    by_docname = {page.docname: page for page in pages}

    llms_text = docs_module.render_llms_document(site_config, pages, expanded=False)
    assert "# pgmpy" in llms_text
    assert "[Getting Started](https://pgmpy.org/started/)" in llms_text
    assert "[Guides](https://pgmpy.org/documentation/)" in llms_text
    assert "[API Reference](https://pgmpy.org/reference/)" in llms_text
    assert "llms-full.txt" in llms_text

    home_structured_data = docs_module.build_structured_data(
        site_config=site_config,
        page=by_docname["index"],
        current_url="https://pgmpy.org/",
        pages=pages,
    )
    assert {item["@type"] for item in home_structured_data} == {"Organization", "WebSite"}

    guide_structured_data = docs_module.build_structured_data(
        site_config=site_config,
        page=by_docname["guides/causal_discovery"],
        current_url="https://pgmpy.org/guides/causal_discovery/",
        pages=pages,
    )
    assert {item["@type"] for item in guide_structured_data} == {"Organization", "BreadcrumbList"}
