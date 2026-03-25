import importlib.util
import json
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
    manifest = {"stable": "v1.2", "releases": ["v1.2", "v1.1"], "development": ["dev"]}

    production = docs_module.resolve_site_config({"PGMPY_DOCS_TARGET": "stable"}, manifest=manifest)
    assert production.base_url == "https://pgmpy.org"
    assert production.is_indexable is True
    assert production.robots_meta == "index,follow,max-image-preview:large"
    assert production.release == "v1.2"

    preview = docs_module.resolve_site_config({"PGMPY_DOCS_TARGET": "preview"}, manifest=manifest)
    assert preview.base_url == "https://pgmpy.org/preview"
    assert preview.is_indexable is False
    assert preview.robots_meta == "noindex,nofollow,noarchive"

    release = docs_module.resolve_site_config(
        {
            "PGMPY_DOCS_TARGET": "v1.1",
        },
        manifest=manifest,
    )
    assert release.base_url == "https://pgmpy.org/v1.1"
    assert release.is_indexable is True
    assert release.robots_meta == "index,follow,max-image-preview:large"
    assert release.version_name == "v1.1"
    assert release.release == "v1.1"
    assert release.version_path == "v1.1"

    legacy_release = docs_module.resolve_site_config(
        {
            "PGMPY_DOCS_ENV": "v1.1",
            "PGMPY_DOCS_RELEASE": "v1.1.2",
            "PGMPY_DOCS_BASEURL": "https://pgmpy.org/v1.1",
        },
        manifest=manifest,
    )
    assert legacy_release.release == "v1.1.2"
    assert legacy_release.base_url == "https://pgmpy.org/v1.1"


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


def test_load_versions_manifest_and_generate_versions_payload(tmp_path):
    docs_module = _load_docs_module()
    manifest_path = tmp_path / "versions.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stable": "v1.2",
                "releases": ["v1.2", "v1.1"],
                "development": ["dev"],
            }
        ),
        encoding="utf-8",
    )

    manifest = docs_module.load_versions_manifest(manifest_path)
    assert manifest["stable"] == "v1.2"
    assert manifest["releases"] == ["v1.2", "v1.1"]
    assert manifest["development"] == ["dev"]

    payload = docs_module.build_versions_payload(
        manifest=manifest,
        site_root_url="https://pgmpy.org",
    )

    assert payload["stable"]["name"] == "stable"
    assert payload["stable"]["label"] == "v1.2 (stable)"
    assert payload["stable"]["url"] == "https://pgmpy.org/"
    assert payload["current_stable"] == "v1.2"
    assert [version["name"] for version in payload["releases"]] == ["v1.2", "v1.1"]
    assert payload["releases"][0]["url"] == "https://pgmpy.org/v1.2/"
    assert payload["in_development"] == [{"name": "dev", "label": "dev", "url": "https://pgmpy.org/dev/"}]


def test_build_versions_payload_requires_stable_release_membership():
    docs_module = _load_docs_module()

    try:
        docs_module.build_versions_payload(
            manifest={"stable": "v1.2", "releases": ["v1.1"], "development": ["dev"]},
            site_root_url="https://pgmpy.org",
        )
    except ValueError as error:
        assert "stable" in str(error)
    else:
        raise AssertionError("Expected build_versions_payload to reject a missing stable release.")
