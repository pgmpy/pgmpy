from pathlib import Path

import pgmpy_docs

if __name__ == "__main__":
    docs_root = Path(__file__).resolve().parents[1]
    pgmpy_docs.stage_docs_sources(docs_root)
