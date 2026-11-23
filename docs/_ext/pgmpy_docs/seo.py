from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .versions import SiteConfig, resolve_site_config

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
class PageRecord:
    docname: str
    title: str
    description: str
    section: str
    source_path: Path


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
    return sorted(
        pages,
        key=lambda page: (
            SECTION_ORDER.get(page.section, 99),
            0 if page.docname in PRIMARY_PAGES else 1,
            page.docname,
        ),
    )


def build_page_url(docname: str, site_config: SiteConfig, target_uri_resolver: Callable[[str], str]) -> str:
    target_uri = target_uri_resolver(docname)
    if not target_uri or target_uri == "/":
        return f"{site_config.base_url}/"
    return f"{site_config.base_url}/{target_uri.lstrip('/')}"


def render_llms_document(
    site_config: SiteConfig,
    pages: list[PageRecord],
    expanded: bool,
    target_uri_resolver: Callable[[str], str],
) -> str:
    sections: dict[str, list[PageRecord]] = {}
    for page in pages:
        is_primary = (
            page.docname in PRIMARY_PAGES or page.docname.startswith("guides/") or page.docname.startswith("api/")
        )
        if not expanded and not is_primary:
            continue
        sections.setdefault(page.section, []).append(page)

    lines = ["# pgmpy", f"> {site_config.default_description}", "", "## Pages"]
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


def build_structured_data(
    site_config: SiteConfig,
    page: PageRecord,
    pages: dict[str, PageRecord],
    target_uri_resolver: Callable[[str], str],
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

    breadcrumb_docnames = ["index"]
    for prefix, parent in (
        ("started/", "started/index"),
        ("guides/", "documentation"),
        ("api/", "reference"),
        ("examples/", "examples"),
    ):
        if page.docname.startswith(prefix):
            breadcrumb_docnames.append(parent)
            break
    if page.docname not in breadcrumb_docnames:
        breadcrumb_docnames.append(page.docname)

    breadcrumbs = []
    for position, crumb_docname in enumerate(breadcrumb_docnames, start=1):
        crumb_page = pages.get(crumb_docname)
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


def on_builder_inited(app: Any) -> None:
    app._pgmpy_docs_site_config = resolve_site_config()
    app._pgmpy_docs_pages = {page.docname: page for page in discover_pages(Path(app.srcdir))}


def on_html_page_context(app: Any, pagename: str, templatename: str, context: dict[str, Any], doctree: Any) -> None:
    if getattr(app.builder, "format", "") != "html":
        return

    site_config = app._pgmpy_docs_site_config
    page = app._pgmpy_docs_pages.get(pagename)
    if page is None:
        page = PageRecord(
            docname=pagename,
            title=context.get("title", "pgmpy"),
            description=site_config.default_description,
            section=_classify_section(pagename),
            source_path=Path(app.srcdir) / Path(app.env.doc2path(pagename, base=False)),
        )

    current_url = build_page_url(pagename, site_config, app.builder.get_target_uri)
    description = page.description or site_config.default_description
    metatags = context.get("metatags", "")
    for marker, tag in (
        ('name="description"', f'<meta name="description" content="{description}" />'),
        ('property="og:description"', f'<meta property="og:description" content="{description}" />'),
        ('name="twitter:card"', '<meta name="twitter:card" content="summary_large_image" />'),
        ('name="twitter:description"', f'<meta name="twitter:description" content="{description}" />'),
        ('property="og:image"', f'<meta property="og:image" content="{site_config.social_image}" />'),
        ('name="twitter:image"', f'<meta name="twitter:image" content="{site_config.social_image}" />'),
        ('name="robots"', f'<meta name="robots" content="{site_config.robots_meta}" />'),
        ('rel="canonical"', f'<link rel="canonical" href="{current_url}" />'),
    ):
        if marker not in metatags:
            metatags += f"\n{tag}"

    context["metatags"] = metatags
    context["pgmpy_current_target_uri"] = app.builder.get_target_uri(pagename)
    context["pgmpy_site_root_url"] = site_config.site_root_url
    context["pgmpy_version_name"] = site_config.version_name
    context["pgmpy_structured_data"] = [
        json.dumps(item, sort_keys=True)
        for item in build_structured_data(
            site_config=site_config,
            page=page,
            pages=app._pgmpy_docs_pages,
            target_uri_resolver=app.builder.get_target_uri,
        )
    ]


def on_build_finished(app: Any, exception: Exception | None) -> None:
    if exception is not None or getattr(app.builder, "format", "") != "html":
        return

    pages = list(app._pgmpy_docs_pages.values())
    outdir = Path(app.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for filename, expanded in (("llms.txt", False), ("llms-full.txt", True)):
        (outdir / filename).write_text(
            render_llms_document(
                app._pgmpy_docs_site_config,
                pages,
                expanded=expanded,
                target_uri_resolver=app.builder.get_target_uri,
            ),
            encoding="utf-8",
        )
