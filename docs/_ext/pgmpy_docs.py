from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SITE_URL = "https://pgmpy.org"
DEFAULT_VERSIONS_FILE = Path(__file__).resolve().parents[1] / "versions.json"
TUTORIALS_PATH_ENV_VAR = "PGMPY_TUTORIALS_PATH"
EXAMPLES_PATH_ENV_VAR = "PGMPY_EXAMPLES_PATH"
DOCS_TARGET_VAR = "PGMPY_DOCS_TARGET"
DOCS_ENV_VAR = "PGMPY_DOCS_ENV"
DOCS_BASEURL_ENV_VAR = "PGMPY_DOCS_BASEURL"
DOCS_SITE_ROOT_ENV_VAR = "PGMPY_DOCS_SITE_ROOT"
DOCS_VERSION_VAR = "PGMPY_DOCS_VERSION"
DOCS_RELEASE_VAR = "PGMPY_DOCS_RELEASE"
DOCS_VERSIONS_FILE_ENV_VAR = "PGMPY_DOCS_VERSIONS_FILE"
RELEASE_VERSION_PATTERN = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)$")
TITLE_UNDERLINE_CHARS = frozenset('=-~^"`:#*+')
SECTION_ORDER = {
    "Home": 0,
    "Getting Started": 1,
    "Guides": 2,
    "Examples": 3,
    "API Reference": 4,
    "Project": 5,
}
PRIMARY_PAGES = {
    "index",
    "started/index",
    "started/install",
    "started/quickstart",
    "documentation",
    "examples",
    "reference",
    "development",
}


@dataclass(frozen=True)
class SiteConfig:
    environment: str
    base_url: str
    site_root_url: str
    is_indexable: bool
    site_name: str
    default_description: str
    social_image: str
    robots_meta: str
    version_name: str
    release: str
    version_path: str


@dataclass(frozen=True)
class PageRecord:
    docname: str
    title: str
    description: str
    section: str
    source_path: Path


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _default_versions_manifest_path() -> Path:
    return DEFAULT_VERSIONS_FILE


def _parse_release_version(version_name: str) -> tuple[int, int] | None:
    match = RELEASE_VERSION_PATTERN.fullmatch(version_name.strip())
    if match is None:
        return None
    return tuple(int(match.group(part)) for part in ("major", "minor"))


def _version_path(version_name: str) -> str:
    if version_name == "stable":
        return ""
    return version_name.strip("/")


def load_versions_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = _default_versions_manifest_path() if path is None else Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "stable": manifest.get("stable"),
        "releases": list(manifest.get("releases", [])),
        "development": list(manifest.get("development", ["dev"])),
    }


def resolve_release_docs_metadata(release_tag: str, manifest: dict[str, Any]) -> dict[str, Any]:
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", release_tag.strip())
    if match is None:
        raise ValueError(f"Unsupported release tag format: {release_tag}")

    docs_version = f"v{match.group(1)}.{match.group(2)}"
    releases = manifest.get("releases", [])
    stable = manifest.get("stable")

    if docs_version not in releases:
        raise ValueError(
            f"Release line {docs_version} is not listed in docs/versions.json. "
            "Update the manifest before building or pushing the release tag."
        )

    return {
        "release_tag": release_tag,
        "docs_version": docs_version,
        "build_stable_root": stable == docs_version,
    }


def _resolved_source_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _replace_tree(
    source: Path,
    destination: Path,
    ignore: Callable[[str, list[str]], set[str]] | None = None,
) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    if source.is_dir():
        shutil.copytree(source, destination, ignore=ignore)


def stage_docs_sources(docs_root: str | Path, environ: dict[str, str] | None = None) -> None:
    """Copy notebook sources into ``docs/`` before a Sphinx build."""

    env = dict(os.environ if environ is None else environ)
    docs_path = Path(docs_root)
    docs_path.mkdir(parents=True, exist_ok=True)

    tutorials_path = _resolved_source_path(
        env.get(TUTORIALS_PATH_ENV_VAR, docs_path.parent.parent / "pgmpy_tutorials" / "notebooks")
    )
    tutorials_assets_root = tutorials_path.parent
    examples_path = _resolved_source_path(env.get(EXAMPLES_PATH_ENV_VAR, docs_path.parent / "examples"))
    ignore_patterns = shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__")

    _replace_tree(tutorials_path, docs_path / "detailed_notebooks", ignore=ignore_patterns)
    for asset_dir in ("csv", "files", "images"):
        _replace_tree(tutorials_assets_root / asset_dir, docs_path / asset_dir, ignore=ignore_patterns)

    _replace_tree(examples_path, docs_path / "examples", ignore=ignore_patterns)


def _resolve_target(env: dict[str, str]) -> str:
    target = env.get(DOCS_TARGET_VAR, "").strip()
    if target:
        return target

    version_name = env.get(DOCS_VERSION_VAR, "").strip()
    if version_name:
        return version_name

    environment = env.get(DOCS_ENV_VAR, "development").strip().lower()
    if environment == "production":
        return "stable"
    if environment in {"development", "dev"}:
        return "dev"
    return environment or "dev"


def _resolve_site_config_from_target(target: str, site_root_url: str, manifest: dict[str, Any]) -> SiteConfig:
    normalized_root = _normalize_base_url(site_root_url)
    development_targets = set(manifest.get("development", ["dev"]))
    stable_version = manifest.get("stable")

    if target == "stable":
        release = stable_version or "stable"
        return SiteConfig(
            environment="production",
            base_url=normalized_root,
            site_root_url=normalized_root,
            is_indexable=True,
            site_name="pgmpy",
            default_description=(
                "pgmpy is a Python library for causal inference, probabilistic modeling, "
                "Bayesian networks, and directed acyclic graphs."
            ),
            social_image=f"{normalized_root}/_static/images/logo.png",
            robots_meta="index,follow,max-image-preview:large",
            version_name="stable",
            release=release,
            version_path="",
        )

    if target in development_targets or target == "preview":
        base_url = f"{normalized_root}/{target}"
        return SiteConfig(
            environment=target,
            base_url=base_url,
            site_root_url=normalized_root,
            is_indexable=False,
            site_name="pgmpy",
            default_description=(
                "pgmpy is a Python library for causal inference, probabilistic modeling, "
                "Bayesian networks, and directed acyclic graphs."
            ),
            social_image=f"{base_url}/_static/images/logo.png",
            robots_meta="noindex,nofollow,noarchive",
            version_name=target,
            release=target,
            version_path=_version_path(target),
        )

    if _parse_release_version(target) is not None:
        base_url = f"{normalized_root}/{target}"
        return SiteConfig(
            environment=target,
            base_url=base_url,
            site_root_url=normalized_root,
            is_indexable=True,
            site_name="pgmpy",
            default_description=(
                "pgmpy is a Python library for causal inference, probabilistic modeling, "
                "Bayesian networks, and directed acyclic graphs."
            ),
            social_image=f"{base_url}/_static/images/logo.png",
            robots_meta="index,follow,max-image-preview:large",
            version_name=target,
            release=target,
            version_path=_version_path(target),
        )

    environment = target or "dev"
    base_url = f"{normalized_root}/{environment}"
    return SiteConfig(
        environment=environment,
        base_url=base_url,
        site_root_url=normalized_root,
        is_indexable=False,
        site_name="pgmpy",
        default_description=(
            "pgmpy is a Python library for causal inference, probabilistic modeling, "
            "Bayesian networks, and directed acyclic graphs."
        ),
        social_image=f"{base_url}/_static/images/logo.png",
        robots_meta="noindex,nofollow,noarchive",
        version_name=environment,
        release=environment,
        version_path=_version_path(environment),
    )


def resolve_site_config(environ: dict[str, str] | None = None, manifest: dict[str, Any] | None = None) -> SiteConfig:
    env = dict(os.environ if environ is None else environ)
    site_root_url = _normalize_base_url(env.get(DOCS_SITE_ROOT_ENV_VAR, DEFAULT_SITE_URL))
    loaded_manifest = load_versions_manifest(env.get(DOCS_VERSIONS_FILE_ENV_VAR)) if manifest is None else manifest
    site_config = _resolve_site_config_from_target(_resolve_target(env), site_root_url, loaded_manifest)

    if DOCS_BASEURL_ENV_VAR in env:
        base_url = _normalize_base_url(env[DOCS_BASEURL_ENV_VAR])
        site_config = SiteConfig(
            environment=site_config.environment,
            base_url=base_url,
            site_root_url=site_config.site_root_url,
            is_indexable=site_config.is_indexable,
            site_name=site_config.site_name,
            default_description=site_config.default_description,
            social_image=f"{base_url}/_static/images/logo.png",
            robots_meta=site_config.robots_meta,
            version_name=site_config.version_name,
            release=env.get(DOCS_RELEASE_VAR, site_config.release).strip() or site_config.release,
            version_path=site_config.version_path,
        )
    elif DOCS_RELEASE_VAR in env:
        site_config = SiteConfig(
            environment=site_config.environment,
            base_url=site_config.base_url,
            site_root_url=site_config.site_root_url,
            is_indexable=site_config.is_indexable,
            site_name=site_config.site_name,
            default_description=site_config.default_description,
            social_image=site_config.social_image,
            robots_meta=site_config.robots_meta,
            version_name=site_config.version_name,
            release=env.get(DOCS_RELEASE_VAR, site_config.release).strip() or site_config.release,
            version_path=site_config.version_path,
        )
    return site_config


def build_versions_payload(manifest: dict[str, Any], site_root_url: str = DEFAULT_SITE_URL) -> dict[str, Any]:
    normalized_root = _normalize_base_url(site_root_url)
    release_versions = list(dict.fromkeys(manifest.get("releases", [])))
    stable_version = manifest.get("stable")

    for version in release_versions:
        if _parse_release_version(version) is None:
            raise ValueError(f"Invalid release version in manifest: {version}")
    if stable_version is not None and stable_version not in release_versions:
        raise ValueError("The manifest stable version must also be present in the releases list.")

    releases = [
        {
            "name": version,
            "label": version,
            "url": f"{normalized_root}/{version}/",
        }
        for version in release_versions
    ]

    stable = None
    if stable_version is not None:
        stable = {
            "name": "stable",
            "label": f"{stable_version} (stable)",
            "url": f"{normalized_root}/",
        }

    return {
        "stable": stable,
        "current_stable": stable_version,
        "releases": releases,
        "in_development": [
            {"name": version, "label": version, "url": f"{normalized_root}/{version}/"}
            for version in dict.fromkeys(manifest.get("development", ["dev"]))
        ],
    }


def build_theme_version_info(site_config: SiteConfig) -> list[dict[str, Any]]:
    return [{"version": site_config.base_url, "title": current_version_label(site_config), "aliases": []}]


def current_version_label(site_config: SiteConfig) -> str:
    if site_config.version_name == "stable" and site_config.release:
        return f"{site_config.release} (stable)"
    return site_config.version_name


def _extract_title_from_rst(text: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines[:-1]):
        stripped = line.strip()
        underline = lines[idx + 1].strip()
        if not stripped or not underline:
            continue
        if set(underline) <= TITLE_UNDERLINE_CHARS and len(underline) >= len(stripped):
            return stripped
    return ""


def _extract_title_from_md(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def _extract_meta_description(text: str) -> str:
    for pattern in (
        r"```{meta}\s*(.*?)```",
        r"\.\. meta::\s*((?:\n\s+:[^:]+:.*)+)",
    ):
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        description_match = re.search(r"^\s*:description:\s*(.+)$", match.group(1), flags=re.MULTILINE)
        if description_match:
            return description_match.group(1).strip()
    return ""


def _extract_first_paragraph_from_rst(text: str) -> str:
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        if line and next_line and set(next_line) <= TITLE_UNDERLINE_CHARS and len(next_line) >= len(line):
            idx += 2
            continue
        if not line:
            idx += 1
            continue
        if line.startswith(":"):
            idx += 1
            continue
        if line.startswith(".. "):
            idx += 1
            while idx < len(lines) and (not lines[idx].strip() or lines[idx].startswith("   ")):
                idx += 1
            continue
        paragraph = [line]
        idx += 1
        while idx < len(lines) and lines[idx].strip():
            paragraph.append(lines[idx].strip())
            idx += 1
        return " ".join(paragraph)
    return ""


def _extract_first_paragraph_from_md(text: str) -> str:
    in_fence = False
    saw_title = False
    paragraph: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not saw_title and line.startswith("# "):
            saw_title = True
            continue
        if not line:
            if paragraph:
                return " ".join(paragraph)
            continue
        if line.startswith(":::") or line.startswith("```{meta}") or line.startswith(":"):
            continue
        if line.startswith("- ") or line.startswith("* "):
            if paragraph:
                return " ".join(paragraph)
            continue
        paragraph.append(line)
    return " ".join(paragraph)


def _classify_section(docname: str) -> str:
    if docname == "index":
        return "Home"
    if docname.startswith("started/"):
        return "Getting Started"
    if docname == "documentation" or docname.startswith("guides/"):
        return "Guides"
    if docname == "examples":
        return "Examples"
    if docname == "reference" or docname.startswith("api/"):
        return "API Reference"
    return "Project"


def _sort_key(page: PageRecord) -> tuple[int, int, str]:
    return (
        SECTION_ORDER.get(page.section, 99),
        0 if page.docname in PRIMARY_PAGES else 1,
        page.docname,
    )


def discover_pages(docs_root: Path) -> list[PageRecord]:
    pages: list[PageRecord] = []
    for path in sorted(docs_root.rglob("*")):
        if path.suffix not in {".rst", ".md"}:
            continue
        if any(part.startswith("_") for part in path.relative_to(docs_root).parts):
            continue
        docname = path.relative_to(docs_root).with_suffix("").as_posix()
        text = path.read_text(encoding="utf-8")
        title = _extract_title_from_md(text) if path.suffix == ".md" else _extract_title_from_rst(text)
        if not title:
            continue
        description = _extract_meta_description(text)
        if not description:
            extractor = _extract_first_paragraph_from_md if path.suffix == ".md" else _extract_first_paragraph_from_rst
            description = extractor(text)
        pages.append(
            PageRecord(
                docname=docname,
                title=title,
                description=description.strip(),
                section=_classify_section(docname),
                source_path=path,
            )
        )
    return sorted(pages, key=_sort_key)


def _pretty_doc_path(docname: str) -> str:
    if docname == "index":
        return "/"
    if docname.endswith("/index"):
        return f"/{docname[:-6]}/"
    return f"/{docname}/"


def _absolute_url(base_url: str, target_uri: str) -> str:
    if not target_uri or target_uri == "/":
        return f"{base_url}/"
    return f"{base_url}/{target_uri.lstrip('/')}"


def build_page_url(
    docname: str,
    site_config: SiteConfig,
    target_uri_resolver: Callable[[str], str] | None = None,
) -> str:
    target_uri = _pretty_doc_path(docname) if target_uri_resolver is None else target_uri_resolver(docname)
    return _absolute_url(site_config.base_url, target_uri)


def _is_primary_llms_page(page: PageRecord) -> bool:
    if page.docname in PRIMARY_PAGES:
        return True
    return page.docname.startswith("guides/") or page.docname.startswith("api/")


def render_llms_document(
    site_config: SiteConfig,
    pages: list[PageRecord],
    expanded: bool,
    target_uri_resolver: Callable[[str], str] | None = None,
) -> str:
    heading = "# pgmpy"
    summary = f"> {site_config.default_description}"
    sections: dict[str, list[PageRecord]] = {}
    for page in pages:
        if not expanded and not _is_primary_llms_page(page):
            continue
        sections.setdefault(page.section, []).append(page)

    lines = [heading, summary, "", "## Pages"]
    for section in ("Home", "Getting Started", "Guides", "Examples", "API Reference", "Project"):
        group = sections.get(section, [])
        if not group:
            continue
        lines.extend(["", f"### {section}"])
        for page in group:
            url = build_page_url(page.docname, site_config, target_uri_resolver)
            lines.append(f"- [{page.title}]({url}): {page.description}")

    lines.extend(
        [
            "",
            "## Machine-readable",
            "",
            f"- [Sitemap]({site_config.base_url}/sitemap.xml): XML sitemap for crawlers and site indexing.",
        ]
    )
    if not expanded:
        lines.append(
            f"- [Full page inventory]({site_config.base_url}/llms-full.txt): "
            "Expanded list of public pages and summaries."
        )
    return "\n".join(lines) + "\n"


def _page_lookup(pages: list[PageRecord]) -> dict[str, PageRecord]:
    return {page.docname: page for page in pages}


def _breadcrumb_docnames(docname: str) -> list[str]:
    if docname == "index":
        return ["index"]

    breadcrumbs = ["index"]
    if docname.startswith("started/"):
        breadcrumbs.append("started/index")
    elif docname.startswith("guides/"):
        breadcrumbs.append("documentation")
    elif docname.startswith("api/"):
        breadcrumbs.append("reference")
    elif docname.startswith("examples/"):
        breadcrumbs.append("examples")

    if docname not in breadcrumbs:
        breadcrumbs.append(docname)
    return breadcrumbs


def build_structured_data(
    site_config: SiteConfig,
    page: PageRecord,
    current_url: str,
    pages: list[PageRecord],
    target_uri_resolver: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    organization_id = f"{site_config.base_url}/#organization"
    website_id = f"{site_config.base_url}/#website"
    graph: list[dict[str, Any]] = [
        {
            "@context": "https://schema.org",
            "@type": "Organization",
            "@id": organization_id,
            "name": site_config.site_name,
            "url": site_config.base_url,
            "logo": site_config.social_image,
            "sameAs": [
                "https://github.com/pgmpy/pgmpy",
                "https://pypi.org/project/pgmpy/",
                "https://www.linkedin.com/company/pgmpy/",
            ],
        }
    ]

    if page.docname == "index":
        graph.append(
            {
                "@context": "https://schema.org",
                "@type": "WebSite",
                "@id": website_id,
                "url": site_config.base_url,
                "name": site_config.site_name,
                "description": site_config.default_description,
                "publisher": {"@id": organization_id},
                "inLanguage": "en",
            }
        )
        return graph

    lookup = _page_lookup(pages)
    breadcrumbs = []
    for position, crumb_docname in enumerate(_breadcrumb_docnames(page.docname), start=1):
        crumb_page = lookup.get(crumb_docname)
        if crumb_page is None:
            continue
        breadcrumbs.append(
            {
                "@type": "ListItem",
                "position": position,
                "name": crumb_page.title,
                "item": build_page_url(crumb_docname, site_config, target_uri_resolver),
            }
        )

    if breadcrumbs:
        graph.append(
            {
                "@context": "https://schema.org",
                "@type": "BreadcrumbList",
                "itemListElement": breadcrumbs,
            }
        )
    return graph


def _append_meta_tag(metatags: str, tag: str, marker: str) -> str:
    if marker in metatags:
        return metatags
    return metatags + tag


def _load_page(app: Any, pagename: str) -> PageRecord | None:
    pages: list[PageRecord] = getattr(app, "_pgmpy_docs_pages", [])
    return _page_lookup(pages).get(pagename)


def on_builder_inited(app: Any) -> None:
    site_config = resolve_site_config()
    pages = discover_pages(Path(app.srcdir))
    setattr(app, "_pgmpy_docs_site_config", site_config)
    setattr(app, "_pgmpy_docs_pages", pages)


def on_html_page_context(app: Any, pagename: str, templatename: str, context: dict[str, Any], doctree: Any) -> None:
    if getattr(app.builder, "format", "") != "html":
        return

    site_config = getattr(app, "_pgmpy_docs_site_config", resolve_site_config())
    page = _load_page(app, pagename)
    if page is None:
        source_path = Path(app.env.doc2path(pagename, base=False))
        full_source_path = Path(app.srcdir) / source_path
        page = PageRecord(
            docname=pagename,
            title=context.get("title", "pgmpy"),
            description=site_config.default_description,
            section=_classify_section(pagename),
            source_path=full_source_path,
        )

    current_target_uri = app.builder.get_target_uri(pagename)
    current_url = build_page_url(pagename, site_config, app.builder.get_target_uri)
    metatags = context.get("metatags", "")
    description = page.description or site_config.default_description
    metatags = _append_meta_tag(
        metatags,
        f'\n<meta name="description" content="{description}" />',
        'name="description"',
    )
    metatags = _append_meta_tag(
        metatags,
        f'\n<meta property="og:description" content="{description}" />',
        'property="og:description"',
    )
    metatags = _append_meta_tag(
        metatags,
        '\n<meta name="twitter:card" content="summary_large_image" />',
        'name="twitter:card"',
    )
    metatags = _append_meta_tag(
        metatags,
        f'\n<meta name="twitter:description" content="{description}" />',
        'name="twitter:description"',
    )
    metatags = _append_meta_tag(
        metatags,
        f'\n<meta property="og:image" content="{site_config.social_image}" />',
        'property="og:image"',
    )
    metatags = _append_meta_tag(
        metatags,
        f'\n<meta name="twitter:image" content="{site_config.social_image}" />',
        'name="twitter:image"',
    )
    metatags = _append_meta_tag(
        metatags,
        f'\n<meta name="robots" content="{site_config.robots_meta}" />',
        'name="robots"',
    )
    metatags = _append_meta_tag(
        metatags,
        f'\n<link rel="canonical" href="{current_url}" />',
        'rel="canonical"',
    )

    context["metatags"] = metatags
    context["pgmpy_current_target_uri"] = current_target_uri
    context["pgmpy_site_root_url"] = site_config.site_root_url
    context["pgmpy_version_name"] = site_config.version_name
    context["pgmpy_structured_data"] = [
        json.dumps(item, sort_keys=True)
        for item in build_structured_data(
            site_config=site_config,
            page=page,
            current_url=current_url,
            pages=getattr(app, "_pgmpy_docs_pages", []),
            target_uri_resolver=app.builder.get_target_uri,
        )
    ]


def on_build_finished(app: Any, exception: Exception | None) -> None:
    if exception is not None or getattr(app.builder, "format", "") != "html":
        return

    site_config = getattr(app, "_pgmpy_docs_site_config", resolve_site_config())
    pages = getattr(app, "_pgmpy_docs_pages", [])
    outdir = Path(app.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "llms.txt").write_text(
        render_llms_document(site_config, pages, expanded=False, target_uri_resolver=app.builder.get_target_uri),
        encoding="utf-8",
    )
    (outdir / "llms-full.txt").write_text(
        render_llms_document(site_config, pages, expanded=True, target_uri_resolver=app.builder.get_target_uri),
        encoding="utf-8",
    )


def setup(app: Any) -> dict[str, Any]:
    app.connect("builder-inited", on_builder_inited)
    app.connect("html-page-context", on_html_page_context)
    app.connect("build-finished", on_build_finished)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
