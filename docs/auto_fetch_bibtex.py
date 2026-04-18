import re
import time
import urllib.parse
from difflib import SequenceMatcher

import requests

BIB_PATH = "docs/references.bib"
CROSSREF_QUERY_URL = "https://api.crossref.org/works?query.bibliographic={query}&rows=3"
CROSSREF_BIBTEX_URL = "https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
MAILTO = "pgmpy-bot@example.com"
REQUEST_TIMEOUT = 10


def clean_text(t):
    return re.sub(r"[{}]", "", t).replace("\n", " ").strip().lower()


def is_good_match(t1, t2):
    t1_c = clean_text(t1)
    t2_c = clean_text(t2)
    if not t1_c or not t2_c:
        return False

    # Substring match (in case local title is truncated)
    if len(t1_c) > 15 and t1_c in t2_c:
        return True
    if len(t2_c) > 15 and t2_c in t1_c:
        return True

    ratio = SequenceMatcher(None, t1_c, t2_c).ratio()
    return ratio > 0.75


def fetch_bibtex_from_crossref(title, authors):
    query = f"{title} {authors}"
    url = CROSSREF_QUERY_URL.format(query=urllib.parse.quote(query))
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": f"pgmpy-fetch/1.0 (mailto:{MAILTO})"})
        r.raise_for_status()
    except Exception as e:
        print(f"  [!] Query failed: {e}")
        return None, None

    data = r.json()
    items = data.get("message", {}).get("items", [])
    if not items:
        return None, None

    # Try to find a good match in top 3
    best_match = None

    for item in items:
        can_title = item.get("title", [""])[0]
        if is_good_match(title, can_title):
            best_match = item
            break

    target_doi = None
    if best_match:
        target_doi = best_match.get("DOI")
    else:
        # Fallback to arxiv or exactly as is if no confident match
        return None, None

    if not target_doi:
        return None, None

    print(f"  -> Found DOI: {target_doi}")

    # Now fetch canonical BibTeX
    bib_url = CROSSREF_BIBTEX_URL.format(doi=target_doi)
    try:
        r = requests.get(
            bib_url,
            headers={"Accept": "application/x-bibtex", "User-Agent": "pgmpy-fetch/1.0"},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        bibtex = r.content.decode("utf-8", errors="ignore").strip()
        return target_doi, bibtex
    except Exception as e:
        print(f"  [!] BibTeX fetch failed: {e}")
        return target_doi, None


def main():
    with open(BIB_PATH, encoding="utf-8") as f:
        raw = f.read()

    # Find all entries
    entries = []
    current_entry = []
    in_entry = False
    brace_depth = 0

    for line in raw.splitlines():
        if not in_entry:
            if line.strip().startswith("@"):
                in_entry = True
                brace_depth = line.count("{") - line.count("}")
                current_entry = [line]
        else:
            current_entry.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                entries.append("\n".join(current_entry))
                in_entry = False
                current_entry = []

    new_entries = []

    for block in entries:
        m_key = re.search(r"@\w+\s*{\s*([^,]+)", block)
        key = m_key.group(1).strip() if m_key else "UNKNOWN"

        m_doi = re.search(r"\bdoi\s*=\s*{([^}]+)}", block, re.IGNORECASE)
        m_title = re.search(r"\btitle\s*=\s*{([^}]+)}", block, re.IGNORECASE)
        m_author = re.search(r"\bauthor\s*=\s*{([^}]+)}", block, re.IGNORECASE)
        m_keyword = re.search(r"\bkeyword\s*=\s*{([^}]+)}", block, re.IGNORECASE)

        # If it has a DOI, keep it as is
        if m_doi:
            new_entries.append(block)
            continue

        if not m_title:
            new_entries.append(block)
            continue

        title = m_title.group(1).replace("\n", " ").strip()
        author = m_author.group(1).replace("\n", " ").strip() if m_author else ""
        keyword = m_keyword.group(1) if m_keyword else "missing"

        title_clean = clean_text(title)
        author_clean = clean_text(author)

        print(f"Fetching for {key}: '{title_clean[:50]}...'")
        time.sleep(0.5)  # polite rate limiting

        doi, canonical_bibtex = fetch_bibtex_from_crossref(title_clean, author_clean)

        if canonical_bibtex and canonical_bibtex.startswith("@"):
            # Replace the bibtex key with our local key and add our keyword constraint
            # Canonical bibtex usually comes back like @article{some_crossref_key, ...}
            # We want to use our original key so docstring citations don't break.
            new_bibtex = re.sub(
                r"^@\w+\s*{[^,]+,",
                lambda m: m.group(0).replace(m.group(0)[m.group(0).find("{") + 1 : -1], key),
                canonical_bibtex,
                count=1,
            )

            # Inject keyword if not present
            if "keyword=" not in new_bibtex.replace(" ", ""):
                # insert it before the last closing brace
                idx = new_bibtex.rfind("}")
                if idx != -1:
                    new_bibtex = new_bibtex[:idx] + f",\n  keyword   = {{{keyword}}}\n" + new_bibtex[idx:]

            new_entries.append(new_bibtex)
        else:
            new_entries.append(block)

    # Write back
    with open(BIB_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(new_entries) + "\n")

    print("Done")


if __name__ == "__main__":
    main()
