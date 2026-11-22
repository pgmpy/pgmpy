from __future__ import annotations

import re
from typing import Any

FOOTCITE_RE = re.compile(r":footcite:[tp]?:`([^`]+)`")


def _bibliography_block(keys: list[str]) -> list[str]:
    """Build a ``.. bibliography::`` block that renders ``keys`` as full citation text.

    ``:filter: False`` means no entry is selected by citation, so only the keys listed as
    directive content are rendered (see sphinxcontrib-bibtex ``get_filtered_entries``): full
    reference text with no citation label, and no ``:cite:``/``:footcite:`` role required.
    One directive per key (with ``:start: continue``) keeps the docstring's own ordering,
    which a single multi-key directive would replace with ``references.bib`` order.
    """
    block: list[str] = []
    for i, key in enumerate(keys):
        block += ["", ".. bibliography::", "   :list: enumerated", "   :filter: False"]
        if i:
            block.append("   :start: continue")
        block += ["", "   " + key]
    block.append("")
    return block


def _is_section_header(lines: list[str], i: int) -> bool:
    # numpydoc section header: a non-empty line underlined by matching '-' chars
    return (
        i + 1 < len(lines)
        and lines[i].strip() != ""
        and lines[i + 1].strip() != ""
        and set(lines[i + 1].strip()) == {"-"}
        and len(lines[i + 1].strip()) >= len(lines[i].strip())
    )


def _references_span(lines: list[str]) -> tuple[int, int] | None:
    """Return the ``(body_start, end)`` line span of the numpydoc "References" section, or None."""
    start = next(
        (i for i, ln in enumerate(lines) if ln.strip() == "References" and _is_section_header(lines, i)),
        None,
    )
    if start is None:
        return None
    end = next((j for j in range(start + 2, len(lines)) if _is_section_header(lines, j)), len(lines))
    return start + 2, end


def expand_references(app: Any, what: str, name: str, obj: Any, options: Any, lines: list[str]) -> None:
    """Render docstring citations as full reference text instead of short labels.

    Docstrings cite works with ``:footcite:t:`` roles, which render only a short ``Author
    [1]`` mark. This turns those into the complete citation text (from ``references.bib``):

    * In a numpydoc "References" section (a bullet list of ``:footcite:`` roles), the bullets
      are replaced by ``.. bibliography::`` directives so the section shows full reference
      text only -- not a label *and* the text.
    * A citation used inline in the prose keeps its inline mark, and its full text is rendered
      in a "References" section (via ``.. footbibliography::``) so inline-cited works also
      appear as full references, not just an author-year mark.

    Runs before numpydoc (lower ``priority``) and edits the raw numpydoc docstring lines.
    """
    if not any(":footcite:" in line for line in lines):
        return

    # Replace the ``:footcite:`` bullets of a "References" section with full-text bibliographies,
    # keeping any non-footcite content (e.g. a plain link bullet) and dropping locator text.
    span = _references_span(lines)
    if span is not None:
        body_start, end = span
        new_body: list[str] = []
        keys: list[str] = []
        insert_at: int | None = None
        for j in range(body_start, end):
            line_keys = FOOTCITE_RE.findall(lines[j])
            if line_keys:
                if insert_at is None:
                    insert_at = len(new_body)
                keys += line_keys
            else:
                new_body.append(lines[j])
        keys = list(dict.fromkeys(keys))  # dedupe, preserving the docstring's citation order
        if keys:
            new_body[insert_at:insert_at] = _bibliography_block(keys)
            lines[body_start:end] = new_body

    # Any ``:footcite:`` left over is an inline (prose) citation: render its full text via a
    # footbibliography so inline-cited works also show up as complete references.
    if any(":footcite:" in line for line in lines):
        span = _references_span(lines)
        if span is None:
            # numpydoc drops a bare trailing directive's blank line, so give it a real section
            lines += ["", "References", "----------", "", ".. footbibliography::", ""]
        else:
            end = span[1]
            lines[end:end] = ["", ".. footbibliography::", ""]


def skip_sklearn_metadata_request_method(
    app: Any,
    what: str,
    name: str,
    obj: Any,
    skip: bool,
    options: Any,
) -> bool | None:
    """Skip scikit-learn-generated metadata request methods in API docs."""
    if getattr(obj, "__module__", None) == "sklearn.utils._metadata_requests" and re.fullmatch(r"set_.+_request", name):
        return True
    return None
