"""
verify_bibtex.py
----------------
Reads docs/references.bib, finds every entry that has a DOI field,
fetches the canonical BibTeX from the CrossRef content-negotiation API,
and compares key bibliographic fields (author, title, year,
journal/booktitle, volume, pages, doi, number).

Usage (from the repo root):
    python docs/verify_bibtex.py

Exit code 0 = all verifiable DOI-bearing entries match.
Exit code 1 = at least one mismatch found.

Entries whose DOI prefix is not indexed by CrossRef (e.g. arXiv 10.48550/*)
are skipped and do not count as failures.

Requirements: requests  (pip install requests)
"""

from __future__ import annotations

import re
import sys
import time
import textwrap
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("'requests' is not installed.  Run: pip install requests")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BIB_FILE = Path(__file__).with_name("references.bib")

# CrossRef content-negotiation endpoint.
# Docs: https://www.crossref.org/documentation/retrieve-metadata/content-negotiation/
CROSSREF_URL = "https://api.crossref.org/works/{doi}/transform/application/x-bibtex"

# CrossRef "polite pool" asks for a mailto in the User-Agent
MAILTO = "pgmpy-dev@example.com"

REQUEST_DELAY_S = 0.5   # polite pause between HTTP requests (seconds)
REQUEST_TIMEOUT = 20    # per-request network timeout (seconds)

# DOI prefixes that CrossRef does not hold records for – skip, don't warn.
SKIP_PREFIXES = ("10.48550",)   # arXiv

# Fields to compare (normalised to lower-case)
COMPARE_FIELDS = {
    "author", "title", "year",
    "journal", "booktitle",
    "volume", "pages", "doi", "number",
}

# ---------------------------------------------------------------------------
# BibTeX parsing
# ---------------------------------------------------------------------------

# --- Local file parser ----------------------------------------------------
# The hand-written .bib file has each entry's closing } on its own line:
#   @article{key,
#     field = {value},
#     ...
#   }
# We rely on \n} as the terminator so nested { } inside values are fine.

_LOCAL_ENTRY_RE = re.compile(
    r"@\w+\s*\{([^,]+),(.*?)\n\}",
    re.DOTALL,
)
_BRACED_FIELD_RE = re.compile(
    r"\b(\w+)\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
    re.DOTALL,
)


def _parse_local_body(body: str) -> dict[str, str]:
    """Extract field→value pairs from a multi-line BibTeX entry body."""
    return {
        m.group(1).lower(): m.group(2).strip()
        for m in _BRACED_FIELD_RE.finditer(body)
    }


def parse_bib(path: Path) -> list[dict]:
    """Return one dict per entry in the local .bib file."""
    raw = path.read_text(encoding="utf-8")
    entries = []
    for m in _LOCAL_ENTRY_RE.finditer(raw):
        key = m.group(1).strip()
        fields: dict[str, str] = {"_key": key}
        fields.update(_parse_local_body(m.group(2)))
        entries.append(fields)
    return entries


# --- CrossRef response parser ---------------------------------------------
# CrossRef returns compact single-line BibTeX, e.g.:
#   @article{Key_2023, title={...}, ..., year={2023}, month=jan }
# The closing } is the LAST character of the string.
# Bare unbraced values (month=jan) are also present.

_BARE_FIELD_RE = re.compile(
    r"\b(\w+)\s*=\s*([^{,}\s][^,}]*?)\s*(?=[,}])",
    re.DOTALL,
)


def _html_decode(text: str) -> str:
    """Decode HTML entities that CrossRef occasionally embeds in BibTeX."""
    return (
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
    )


def _parse_crossref_body(body: str) -> dict[str, str]:
    """
    Extract field→value pairs from a CrossRef BibTeX body.
    Braced values take priority; bare values fill the rest.
    """
    fields: dict[str, str] = {}
    # Braced values first
    for m in _BRACED_FIELD_RE.finditer(body):
        fields[m.group(1).lower()] = m.group(2).strip()
    # Bare values for anything not yet captured
    for m in _BARE_FIELD_RE.finditer(body):
        name = m.group(1).lower()
        if name not in fields:
            fields[name] = m.group(2).strip()
    return fields


def _parse_crossref_bibtex(raw: str) -> dict | None:
    """
    Parse a CrossRef BibTeX string.  Uses brace-depth counting to find the
    true end of the entry rather than a regex terminator.
    Returns field dict, or None if the string cannot be parsed.
    """
    # Find the opening @type{ position
    at_pos = raw.find("@")
    if at_pos == -1:
        return None

    # Walk forward to find the matching closing } for the entry
    depth = 0
    start = None
    body_start = None
    comma_seen = False
    for i, ch in enumerate(raw[at_pos:], start=at_pos):
        if ch == "{":
            depth += 1
            if depth == 1:
                start = i          # opening brace of the entry
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                # i is the index of the matching closing brace
                body = raw[body_start:i] if body_start is not None else ""
                return _parse_crossref_body(_html_decode(body))
        elif ch == "," and depth == 1 and not comma_seen:
            # first comma after the key → body starts here
            body_start = i + 1
            comma_seen = True

    return None   # unmatched braces


# ---------------------------------------------------------------------------
# Normalisation for comparison
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """
    Normalise a BibTeX field value for fuzzy comparison:
      - decode HTML entities and numeric dash entities
      - strip LaTeX brace groups e.g. {Bayesian} -> Bayesian
      - collapse ALL Unicode/ASCII dash variants to plain '-'
      - normalise LaTeX \\& to plain &
      - strip trademark/registered/copyright symbols
      - collapse whitespace, lower-case
    """
    text = _html_decode(text)
    # Numeric HTML entities for dashes that _html_decode doesn't cover
    text = re.sub(r'&#(?:8211|8212|45);', '-', text)
    text = re.sub(r"\{([^{}]+)\}", r"\1", text)
    # Comprehensive Unicode dash normalisation – covers en-dash (U+2013),
    # em-dash (U+2014), figure-dash (U+2012), minus (U+2212), soft-hyphen
    # (U+00AD), and several others CrossRef or LaTeX might produce.
    text = re.sub(
        r'[\u00AD\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]',
        '-', text
    )
    text = re.sub(r'-{2,}', '-', text)       # -- or --- -> single -
    text = text.replace('\\&', '&')          # LaTeX \& -> &
    # Strip ® © ™ that CrossRef embeds in some journal names
    text = re.sub(r'[\u00ae\u00a9\u2122]', '', text)
    # Normalise spaces after initials:  'R. C.' -> 'R.C.'
    text = re.sub(r'\.\s+([A-Z]\.)', r'.\1', text)
    text = ' '.join(text.split())
    return text.lower().strip()


# ---------------------------------------------------------------------------
# CrossRef fetch
# ---------------------------------------------------------------------------

def fetch_canonical(doi: str) -> dict:
    """
    Fetch canonical BibTeX for *doi* via CrossRef content-negotiation.
    Returns a dict of field values, or {"_error": "..."} on failure.
    """
    url = CROSSREF_URL.format(doi=doi)
    try:
        r = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "Accept": "application/x-bibtex",
                "User-Agent": f"pgmpy-bibtex-verifier/1.0 (mailto:{MAILTO})",
            },
            allow_redirects=True,
        )
        r.raise_for_status()
    except requests.HTTPError as exc:
        return {"_error": f"HTTP {exc.response.status_code} for DOI {doi}"}
    except requests.RequestException as exc:
        return {"_error": str(exc)}

    # Force UTF-8: CrossRef's Content-Type may not declare utf-8, causing
    # requests to fall back to Latin-1 and turning – (E2 80 93) into â€"
    raw = r.content.decode("utf-8", errors="replace").strip()
    if not raw.startswith("@"):
        return {"_error": f"Unexpected response: {raw[:120]}"}

    fields = _parse_crossref_bibtex(raw)
    if fields is None:
        return {"_error": f"Could not parse BibTeX for DOI {doi}:\n{raw[:400]}"}

    # CrossRef returns a stub entry when a DOI has been deleted/retracted.
    # The journal field will read 'Crossref Listing of Deleted DOIs'.
    journal = fields.get("journal", "")
    if "deleted" in _norm(journal) and "crossref" in _norm(journal):
        return {"_error": (
            f"DOI {doi} is listed as deleted in CrossRef "
            "(DOI may have been retired or reassigned; entry cannot be verified)"
        )}

    return fields


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_entries(local: dict, canonical: dict) -> list[str]:
    """
    Return human-readable mismatch strings for every field that differs.
    Empty list means the entries agree on all checked fields.

    Special cases handled:
    - journal/booktitle swap: CrossRef indexes some @inproceedings papers as
      @article, putting the venue in 'journal' instead of 'booktitle'.
      We accept a match if the local value appears in either canonical field.
    - pages present only in local: CrossRef sometimes lacks page numbers for
      older papers (e.g. Annals of Statistics).  Not counted as a mismatch.
    """
    diffs = []
    loc_journal   = _norm(local.get('journal',   ''))
    loc_booktitle = _norm(local.get('booktitle', ''))
    can_journal   = _norm(canonical.get('journal',   ''))
    can_booktitle = _norm(canonical.get('booktitle', ''))

    for field in sorted(COMPARE_FIELDS):
        loc_val = _norm(local.get(field, ''))
        can_val = _norm(canonical.get(field, ''))

        # journal/booktitle symmetry – handle CrossRef article/inproceedings quirk
        if field in ('journal', 'booktitle'):
            can_venue = can_journal or can_booktitle
            loc_venue = loc_journal or loc_booktitle
            if loc_val and loc_val == can_venue:
                continue
            if can_val and can_val == loc_venue:
                continue

        # Pages absent on CrossRef side only – not a data error in our file
        if field == 'pages' and loc_val and not can_val:
            continue

        if not loc_val and not can_val:
            continue   # both absent
        if loc_val != can_val:
            diffs.append(
                f"  Field '{field}':\n"
                f"    LOCAL    : {loc_val or '(missing)'}\n"
                f"    CANONICAL: {can_val or '(missing)'}"
            )
    return diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not BIB_FILE.exists():
        print(f"[ERROR] BibTeX file not found: {BIB_FILE}")
        return 1

    entries = parse_bib(BIB_FILE)
    doi_entries = [e for e in entries if "doi" in e]

    checkable = [e for e in doi_entries
                 if not e["doi"].strip().startswith(SKIP_PREFIXES)]
    skipped   = [e for e in doi_entries
                 if e["doi"].strip().startswith(SKIP_PREFIXES)]

    print(f"[INFO] Parsed  : {len(entries)} total entries from {BIB_FILE.name}")
    print(f"[INFO] DOI entries : {len(doi_entries)}")
    print(f"[INFO]   To verify : {len(checkable)}")
    print(f"[INFO]   Skipped   : {len(skipped)}  "
          f"(prefixes not in CrossRef: {SKIP_PREFIXES})")
    print(f"[INFO] Querying CrossRef API...\n")

    errors:   list[str] = []
    warnings: list[str] = []

    for i, entry in enumerate(checkable, 1):
        key = entry["_key"]
        doi = entry["doi"].strip()
        print(f"[{i:>3}/{len(checkable)}]  {key}")
        print(f"          DOI: {doi}")

        canonical = fetch_canonical(doi)
        time.sleep(REQUEST_DELAY_S)

        if "_error" in canonical:
            msg = f"[WARN] Could not fetch '{key}': {canonical['_error']}"
            print(f"       {msg}")
            warnings.append(msg)
            continue

        diffs = compare_entries(entry, canonical)
        if diffs:
            block = "\n".join(diffs)
            msg = f"[FAIL] MISMATCH in '{key}':\n{block}"
            print(msg)
            errors.append(msg)
        else:
            print("       [ OK ]")

    # Summary -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total entries in file    : {len(entries)}")
    print(f"  Entries with DOI         : {len(doi_entries)}")
    print(f"  Checked via CrossRef     : {len(checkable)}")
    print(f"  Skipped (arXiv etc.)     : {len(skipped)}")
    print(f"  Mismatches found         : {len(errors)}")
    print(f"  Could-not-verify (warns) : {len(warnings)}")

    if skipped:
        print("\n--- SKIPPED (no CrossRef record expected) ---")
        for e in skipped:
            print(f"  {e['_key']}  (doi: {e['doi'].strip()})")

    if warnings:
        print("\n--- FETCH WARNINGS (not failures) ---")
        for w in warnings:
            print(f"  {w}")

    if errors:
        print("\n--- MISMATCHES ---")
        for e in errors:
            print(textwrap.indent(e, "  "))
        print("\n[FAIL] Verification FAILED - fix the mismatches listed above.")
        return 1

    if warnings and not errors:
        print("\n[PASS] No mismatches found."
              " Some entries could not be fetched (see warnings above).")
        return 0

    print("\n[PASS] All checkable DOI-bearing entries verified successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
