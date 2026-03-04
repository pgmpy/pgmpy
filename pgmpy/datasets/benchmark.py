"""On-demand loaders for standard Bayesian Network benchmark datasets."""

from __future__ import annotations

import gzip
import hashlib
import os
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen

import pandas as pd

from pgmpy.global_vars import PGMPY_DATA_HOME
from pgmpy.readwrite import BIFReader


@dataclass(frozen=True)
class _BenchmarkSource:
    name: str
    source_url: str
    checksum_sha256: str
    citation: str
    license: str
    num_nodes: int
    num_edges: int
    variables: tuple[str, ...]
    description: str


_BENCHMARK_SOURCES = {
    "alarm": _BenchmarkSource(
        name="alarm",
        source_url="https://raw.githubusercontent.com/pgmpy/example_models/main/discrete/alarm.bif.gz",
        checksum_sha256="f5860caa7137b5817e0c4f169cce0f1451ce6c246b96f5ce4fb4a8167993be29",
        citation=(
            "I. A. Beinlich, H. J. Suermondt, R. M. Chavez, and G. F. Cooper, "
            "'The ALARM Monitoring System: A Case Study with Two Probabilistic Inference Techniques "
            "for Belief Networks,' in AIME 1989."
        ),
        license="CC-BY 4.0",
        num_nodes=37,
        num_edges=46,
        variables=(
            "HISTORY",
            "CVP",
            "PCWP",
            "HYPOVOLEMIA",
            "LVEDVOLUME",
            "LVFAILURE",
            "STROKEVOLUME",
            "ERRLOWOUTPUT",
            "HRBP",
            "HREKG",
            "ERRCAUTER",
            "HRSAT",
            "INSUFFANESTH",
            "ANAPHYLAXIS",
            "TPR",
            "EXPCO2",
            "KINKEDTUBE",
            "MINVOL",
            "FIO2",
            "PVSAT",
            "SAO2",
            "PAP",
            "PULMEMBOLUS",
            "SHUNT",
            "INTUBATION",
            "PRESS",
            "DISCONNECT",
            "MINVOLSET",
            "VENTMACH",
            "VENTTUBE",
            "VENTLUNG",
            "VENTALV",
            "ARTCO2",
            "CATECHOL",
            "HR",
            "CO",
            "BP",
        ),
        description=(
            "ALARM is a 37-node benchmark Bayesian Network for intensive-care monitoring "
            "and probabilistic inference evaluation."
        ),
    ),
    "asia": _BenchmarkSource(
        name="asia",
        source_url="https://raw.githubusercontent.com/pgmpy/example_models/main/discrete/asia.bif.gz",
        checksum_sha256="7d5e7b8549834824c1b05f367eb7860fa3fca3adaf3649f0c7be6035af197197",
        citation=(
            "S. L. Lauritzen and D. J. Spiegelhalter, 'Local Computations with Probabilities on "
            "Graphical Structures and Their Application to Expert Systems,' JRSS B, 1988."
        ),
        license="CC-BY 4.0",
        num_nodes=8,
        num_edges=8,
        variables=("asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"),
        description=(
            "Asia (Lung Cancer) is a compact 8-node benchmark Bayesian Network used for "
            "structure learning and exact inference demonstrations."
        ),
    ),
}


def _normalize_seed(
    sample_id: int | str | None = None,
    random_state: int | None = None,
) -> int | None:
    if random_state is not None:
        if isinstance(random_state, bool) or random_state < 0:
            raise ValueError(
                f"random_state must be a non-negative integer or None. Got: {random_state}"
            )
        return random_state

    if sample_id is None:
        return None

    if isinstance(sample_id, int):
        if isinstance(sample_id, bool) or sample_id < 0:
            raise ValueError(
                f"sample_id must be a non-negative integer, a string, or None. Got: {sample_id}"
            )
        return sample_id

    if not isinstance(sample_id, str):
        raise ValueError(
            f"sample_id must be a non-negative integer, a string, or None. Got type: {type(sample_id).__name__}"
        )

    return int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:8], 16)


def _get_cache_path(dataset_name: str) -> str:
    return os.path.join(PGMPY_DATA_HOME, "benchmark_datasets", f"{dataset_name}.bif.gz")


def _download_if_needed(dataset_name: str, force_download: bool = False) -> bytes:
    source = _BENCHMARK_SOURCES[dataset_name]
    cache_path = _get_cache_path(dataset_name)

    if os.path.exists(cache_path) and not force_download:
        with open(cache_path, "rb") as f:
            return f.read()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with urlopen(source.source_url, timeout=60) as response:
        raw_data = response.read()

    checksum = hashlib.sha256(raw_data).hexdigest()
    if checksum != source.checksum_sha256:
        raise ValueError(
            f"Checksum mismatch while downloading '{dataset_name}'. "
            f"Expected: {source.checksum_sha256}; got: {checksum}. "
            "If this persists, retry with force_download=True."
        )

    with open(cache_path, "wb") as f:
        f.write(raw_data)

    return raw_data


def _load_model(dataset_name: str, force_download: bool = False):
    raw_data = _download_if_needed(dataset_name=dataset_name, force_download=force_download)
    bif = gzip.decompress(raw_data).decode("utf-8")
    return BIFReader(string=bif).get_model()


def _load_benchmark_dataset(
    dataset_name: str,
    n_samples: int,
    sample_id: int | str | None = None,
    random_state: int | None = None,
    force_download: bool = False,
    return_model: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, object]:
    if n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer. Got: {n_samples}")

    model = _load_model(dataset_name=dataset_name, force_download=force_download)
    seed = _normalize_seed(sample_id=sample_id, random_state=random_state)
    data = model.simulate(n_samples=n_samples, seed=seed, show_progress=False)

    if return_model:
        return data, model

    return data


def load_alarm(
    n_samples: int = 10000,
    sample_id: int | str | None = None,
    random_state: int | None = None,
    force_download: bool = False,
    return_model: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, object]:
    """Load sampled data from the ALARM Bayesian Network benchmark.

    The model definition is downloaded on first use and cached under pgmpy's data
    home. Data samples are then generated from the parameterized benchmark model.

    Parameters
    ----------
    n_samples : int, default=10000
        Number of rows to sample.

    sample_id : int | str | None, default=None
        Stable sample identifier used to derive the sampling seed when
        `random_state` is not provided.

    random_state : int | None, default=None
        Explicit sampling seed. If provided, takes precedence over `sample_id`.

    force_download : bool, default=False
        If True, redownloads the model file even if a cached copy exists.

    return_model : bool, default=False
        If True, returns `(data, model)`.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, DiscreteBayesianNetwork]

    References
    ----------
    - I. A. Beinlich, H. J. Suermondt, R. M. Chavez, and G. F. Cooper,
      "The ALARM Monitoring System: A Case Study with Two Probabilistic Inference
      Techniques for Belief Networks," in AIME 1989.
    - Source model: https://github.com/pgmpy/example_models
    - License: CC-BY 4.0

    """
    return _load_benchmark_dataset(
        dataset_name="alarm",
        n_samples=n_samples,
        sample_id=sample_id,
        random_state=random_state,
        force_download=force_download,
        return_model=return_model,
    )


def load_asia(
    n_samples: int = 10000,
    sample_id: int | str | None = None,
    random_state: int | None = None,
    force_download: bool = False,
    return_model: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, object]:
    """Load sampled data from the Asia Bayesian Network benchmark.

    The model definition is downloaded on first use and cached under pgmpy's data
    home. Data samples are then generated from the parameterized benchmark model.

    Parameters
    ----------
    n_samples : int, default=10000
        Number of rows to sample.

    sample_id : int | str | None, default=None
        Stable sample identifier used to derive the sampling seed when
        `random_state` is not provided.

    random_state : int | None, default=None
        Explicit sampling seed. If provided, takes precedence over `sample_id`.

    force_download : bool, default=False
        If True, redownloads the model file even if a cached copy exists.

    return_model : bool, default=False
        If True, returns `(data, model)`.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, DiscreteBayesianNetwork]

    References
    ----------
    - S. L. Lauritzen and D. J. Spiegelhalter,
      "Local Computations with Probabilities on Graphical Structures and Their
      Application to Expert Systems," JRSS B, 1988.
    - Source model: https://github.com/pgmpy/example_models
    - License: CC-BY 4.0

    """
    return _load_benchmark_dataset(
        dataset_name="asia",
        n_samples=n_samples,
        sample_id=sample_id,
        random_state=random_state,
        force_download=force_download,
        return_model=return_model,
    )


def get_benchmark_metadata(name: str) -> dict[str, Any]:
    """Return source, citation, and license metadata for a benchmark dataset.

    Parameters
    ----------
    name : {'alarm', 'asia'}
        Benchmark dataset name.

    Returns
    -------
    dict
        Keys: `name`, `source_url`, `citation`, `license`, `num_nodes`,
        `num_edges`, `variables`, `description`.

    """
    if name not in _BENCHMARK_SOURCES:
        raise ValueError(
            f"Unknown benchmark dataset: {name}. Available datasets: {list(_BENCHMARK_SOURCES.keys())}"
        )

    source = _BENCHMARK_SOURCES[name]
    return {
        "name": source.name,
        "source_url": source.source_url,
        "citation": source.citation,
        "license": source.license,
        "num_nodes": source.num_nodes,
        "num_edges": source.num_edges,
        "variables": source.variables,
        "description": source.description,
    }
