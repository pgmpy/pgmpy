from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SITE_URL = "https://pgmpy.org"
DEFAULT_VERSIONS_FILE = Path(__file__).resolve().parents[2] / "versions.json"
DEFAULT_DESCRIPTION = (
    "pgmpy is a Python library for causal inference, probabilistic modeling, "
    "Bayesian networks, and directed acyclic graphs."
)
DOCS_TARGET_VAR = "PGMPY_DOCS_TARGET"
DOCS_ENV_VAR = "PGMPY_DOCS_ENV"
DOCS_BASEURL_ENV_VAR = "PGMPY_DOCS_BASEURL"
DOCS_SITE_ROOT_ENV_VAR = "PGMPY_DOCS_SITE_ROOT"
DOCS_VERSION_VAR = "PGMPY_DOCS_VERSION"
DOCS_RELEASE_VAR = "PGMPY_DOCS_RELEASE"
DOCS_VERSIONS_FILE_ENV_VAR = "PGMPY_DOCS_VERSIONS_FILE"
RELEASE_VERSION_PATTERN = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)$")
INDEXABLE_ROBOTS_META = "index,follow,max-image-preview:large"
NOT_INDEXABLE_ROBOTS_META = "noindex,nofollow,noarchive"


@dataclass(frozen=True)
class SiteConfig:
    base_url: str
    site_root_url: str
    robots_meta: str
    version_name: str
    release: str
    site_name: str = "pgmpy"
    default_description: str = DEFAULT_DESCRIPTION

    @property
    def social_image(self) -> str:
        return f"{self.base_url}/_static/images/logo.png"


def load_versions_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = DEFAULT_VERSIONS_FILE if path is None else Path(path)
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


def resolve_site_config(environ: dict[str, str] | None = None, manifest: dict[str, Any] | None = None) -> SiteConfig:
    env = dict(os.environ if environ is None else environ)
    loaded_manifest = load_versions_manifest(env.get(DOCS_VERSIONS_FILE_ENV_VAR)) if manifest is None else manifest
    site_root_url = env.get(DOCS_SITE_ROOT_ENV_VAR, DEFAULT_SITE_URL).rstrip("/")

    # Step 1: The target is an explicit docs target, else a version name, else derived from the environment.
    target = env.get(DOCS_TARGET_VAR, "").strip() or env.get(DOCS_VERSION_VAR, "").strip()
    if not target:
        environment = env.get(DOCS_ENV_VAR, "development").strip().lower()
        target = {"production": "stable", "development": "dev", "dev": "dev"}.get(environment, environment or "dev")

    # Step 2: Stable is served from the site root; every other target from a subdirectory. Only stable and
    # release lines that are not under development are indexable.
    if target == "stable":
        base_url = site_root_url
        release = loaded_manifest.get("stable") or "stable"
        is_indexable = True
    else:
        base_url = f"{site_root_url}/{target}"
        release = target
        is_indexable = bool(RELEASE_VERSION_PATTERN.fullmatch(target)) and target not in loaded_manifest.get(
            "development", ["dev"]
        )

    # Step 3: Environment overrides for the base URL and the release label.
    return SiteConfig(
        base_url=env[DOCS_BASEURL_ENV_VAR].rstrip("/") if DOCS_BASEURL_ENV_VAR in env else base_url,
        site_root_url=site_root_url,
        robots_meta=INDEXABLE_ROBOTS_META if is_indexable else NOT_INDEXABLE_ROBOTS_META,
        version_name=target,
        release=env.get(DOCS_RELEASE_VAR, "").strip() or release,
    )


def build_versions_payload(manifest: dict[str, Any], site_root_url: str = DEFAULT_SITE_URL) -> dict[str, Any]:
    normalized_root = site_root_url.rstrip("/")
    release_versions = list(dict.fromkeys(manifest.get("releases", [])))
    stable_version = manifest.get("stable")

    for version in release_versions:
        if RELEASE_VERSION_PATTERN.fullmatch(version.strip()) is None:
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


def current_version_label(site_config: SiteConfig) -> str:
    if site_config.version_name == "stable" and site_config.release:
        return f"{site_config.release} (stable)"
    return site_config.version_name
