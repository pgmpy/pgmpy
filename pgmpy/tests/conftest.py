"""Shared pytest fixtures for the pgmpy test suite.

Example models and datasets are normally fetched from Hugging Face through ``pgmpy.utils.hf_hub.read_hf_file``. To keep
the test suite self-contained and free of network access, the autouse fixture below redirects that single byte-fetch
seam to local fixture files committed under ``test_data/``. If a test requests an asset that is not committed locally,
the redirect raises a clear ``FileNotFoundError`` instead of silently going online, so any new network dependency fails
loudly.
"""

from pathlib import Path

import pytest

import pgmpy.datasets._base as _datasets_base
import pgmpy.example_models._base as _example_models_base
import pgmpy.utils.hf_hub as _hf_hub

TEST_DATA_DIR = Path(__file__).parent / "test_data"

# Maps the Hugging Face ``repo_id`` to the local sub-directory holding its assets.
_REPO_DIRS = {
    "pgmpy/example_models": "example_models",
    "pgmpy/example_datasets": "example_datasets",
}


def _local_read_hf_file(*, repo_id, filename, repo_type=None, revision="main"):
    """Local stand-in for ``read_hf_file`` that reads from ``test_data/``."""
    subdir = _REPO_DIRS.get(repo_id)
    if subdir is None:
        raise FileNotFoundError(
            f"No local test fixtures registered for repo_id={repo_id!r}. "
            f"Add the asset under {TEST_DATA_DIR} to keep tests offline."
        )

    path = TEST_DATA_DIR / subdir / filename.lstrip("/")
    if not path.is_file():
        raise FileNotFoundError(
            f"Local test fixture not found: {path} (repo_id={repo_id!r}, filename={filename!r}). "
            f"Commit it under {TEST_DATA_DIR} so the test suite stays offline."
        )
    return path.read_bytes()


@pytest.fixture(autouse=True)
def offline_example_assets(monkeypatch):
    """Serve example models/datasets from local files instead of Hugging Face.

    ``read_hf_file`` is imported into the ``_base`` modules at import time, so it
    is patched both at its source and at each call site.
    """
    monkeypatch.setattr(_hf_hub, "read_hf_file", _local_read_hf_file)
    monkeypatch.setattr(_example_models_base, "read_hf_file", _local_read_hf_file)
    monkeypatch.setattr(_datasets_base, "read_hf_file", _local_read_hf_file)
