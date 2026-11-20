from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

TUTORIALS_PATH_ENV_VAR = "PGMPY_TUTORIALS_PATH"
EXAMPLES_PATH_ENV_VAR = "PGMPY_EXAMPLES_PATH"


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
