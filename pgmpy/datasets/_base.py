from __future__ import annotations

import hashlib
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge
from pgmpy.global_vars import PGMPY_DATA_HOME
from pgmpy.utils._safe_import import _safe_import

requests = _safe_import("requests")


@dataclass
class Dataset:
    name: str
    data: pd.DataFrame
    expert_knowledge: Optional[ExpertKnowledge] = None
    ground_truth: Optional[DAG] = None
    tags: Dict[str, Any] = None

    def __str__(self) -> str:
        return (
            f"Dataset(name={self.name}, \n data=DataFrame of size: {self.data.shape}, \n "
            f"expert_knowledge={self.expert_knowledge}, \n ground_truth={self.ground_truth}, \n tags={self.tags})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class _BaseDataset:
    @staticmethod
    def _parse_expert_knowledge(raw_expert_knowledge: bytes) -> ExpertKnowledge:
        """
        Helper method to parse expert knowledge from raw bytes.
        """
        text = raw_expert_knowledge.decode("utf-8-sig", errors="ignore")

        temporal: List[List[str]] = []
        forbids: List[Tuple[str, str]] = []
        requires: List[Tuple[str, str]] = []

        section = None

        for raw_line in text.splitlines():
            stripped = raw_line.strip()

            if not stripped:
                continue

            lower = stripped.lower()

            # Section headers
            if lower == "/knowledge":
                section = None
                continue
            if lower == "addtemporal":
                section = "addtemporal"
                continue
            if lower == "forbiddirect":
                section = "forbiddirect"
                continue
            if lower in ("requiredirect", "requireddirect"):
                section = "requiredirect"
                continue

            # Content, depending on current section
            if section == "addtemporal":
                # Treat lines that are just an integer as placeholders for empty lines
                if stripped.isdigit():
                    temporal.append([])
                else:
                    tokens = stripped.split()
                    # Skip the first token; its the line number
                    temporal.append(tokens[1:])

            elif section == "forbiddirect":
                tokens = stripped.split()
                forbids.append((tokens[0], tokens[1]))

            elif section == "requiredirect":
                tokens = stripped.split()
                requires.append((tokens[0], tokens[1]))

        return ExpertKnowledge(
            forbidden_edges=forbids, required_edges=requires, temporal_order=temporal
        )

    @classmethod
    def _get_raw_data(cls, data_type, url) -> bytes:
        """
        Checks if the data is cached locally; if not, fetches it from the URL and caches it.
        """
        cache_dir_path = os.path.join(
            PGMPY_DATA_HOME,
            hashlib.sha256(f"{cls.name}_{cls.base_url}".encode()).hexdigest(),
        )

        path = os.path.join(cache_dir_path, data_type)

        if os.path.exists(path):
            with open(path, "rb") as f:
                raw_data = f.read()
        else:
            os.makedirs(cache_dir_path, exist_ok=True)
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            raw_data = resp.content
            with open(path, "wb") as f:
                f.write(raw_data)
        return raw_data

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        """Fetches/reads from cache the data associated with the dataset."""
        raw_data = cls._get_raw_data("data", cls.data_url)
        df = pd.read_csv(io.BytesIO(raw_data), sep="\t")
        if cls.tags.get("has_missing_data"):
            df.replace(cls.missing_values_marker, pd.NA, inplace=True)
        return df

    @classmethod
    def load_expert_knowledge(cls) -> ExpertKnowledge:
        """Fetches/reads from cache the expert knowledge associated with the dataset."""
        if not cls.tags.get("has_expert_knowledge"):
            return None

        raw_data = cls._get_raw_data("expert_knowledge", cls.expert_knowledge_url)
        expert_knowledge = cls._parse_expert_knowledge(raw_data)
        return expert_knowledge

    @classmethod
    def load_ground_truth(cls) -> DAG:
        """Fetches/reads from cache the ground truth DAG associated with the dataset."""
        if not cls.tags.get("has_ground_truth"):
            return None

        raw_data = cls._get_raw_data("ground_truth", cls.ground_truth_url).decode(
            "utf-8-sig", errors="ignore"
        )
        return DAG.from_dagitty(raw_data)

    @staticmethod
    def clear_cache() -> None:
        """
        Clears the cached data for all datasets.
        """
        if os.path.exists(PGMPY_DATA_HOME):
            Path.rmdir(PGMPY_DATA_HOME)


class _CovarianceMixin:
    """
    This mixin class provides functionality to load datasets defined by a covariance matrix. Mainly the `load_dataframe`
    method is overridden to generate data from the covariance matrix instead of loading a static data file as is the
    case with `_BaseDataset`.
    """

    @classmethod
    def _load_covariance_matrix(cls) -> pd.DataFrame:
        """
        Fetches the data and creates a covariance matrix DataFrame.
        """
        raw_data = cls._get_raw_data("covariance_matrix", cls.data_url).decode(
            "utf-8-sig", errors="ignore"
        )

        lines = raw_data.strip().splitlines()
        # First replace multiple spaces with a single space and then split the line on either \t or space. Datasets are
        # not uniform.
        names = re.split(r"\t|\ ", re.sub(r"\s+", " ", lines[1].strip()))

        mat = np.zeros((len(names), len(names)), dtype=float)

        for i, line in enumerate(lines[2 : 2 + len(names)]):
            vals = np.fromstring(line, sep="\t", dtype=float)
            mat[i, : i + 1] = vals
            mat[: i + 1, i] = vals

        return pd.DataFrame(mat, columns=names, index=names)

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        """Method to create data from covariance matrix. When the `_CovarDatasetMixin is
        used this method is supposed to override the _BaseDataset.load_dataframe method.

        ** Hence, when using this mixin, _CovarDatasetMixin should be the first parent class. **
        """
        cov_matrix = cls._load_covariance_matrix()
        mean = [0] * cls.tags["n_variables"]
        data = pd.DataFrame(
            np.random.multivariate_normal(
                mean, cov_matrix.values, size=cls.tags["n_samples"]
            ),
            columns=cov_matrix.columns,
        )
        return data


class _DatasetRegistry:
    """
    Registry for dataset classes.

    Example
    -------
    >>> from pgmpy.datasets import DATASET_REGISTRY
    >>> all_datasets = DATASET_REGISTRY.list_datasets()
    >>> filtered_datasets = DATASET_REGISTRY.list_datasets(
    ...     has_ground_truth=True, is_mixed=True
    ... )

    """

    _REQUIRED_TAGS = {
        "n_variables",
        "n_samples",
        "has_ground_truth",
        "has_expert_knowledge",
        "has_missing_data",
        "is_simulated",
        "is_interventional",
        "is_discrete",
        "is_continuous",
        "is_mixed",
        "is_ordinal",
    }

    def __init__(self) -> None:
        self._by_name: Dict[str, Type["_BaseDataset"]] = {}
        self._by_tag: Dict[Tuple[str, Any], Set[str]] = {}

    def register(self, cls: Type["_BaseDataset"]) -> None:
        # Step 1: Check if the name is defined.
        if not hasattr(cls, "name"):
            raise TypeError("Dataset classes must define a string 'name' attribute.")

        # Step 2: Check if all required tags are defined.
        if not hasattr(cls, "tags"):
            raise TypeError("Dataset classes must define a 'tags' attribute as a dict.")
        else:
            missing_tags = self._REQUIRED_TAGS - cls.tags.keys()
            if missing_tags:
                raise ValueError(
                    f"Dataset '{cls.__name__}' is missing required tags: {missing_tags}"
                )
        # Step 3: Check if all required attributes/URLs are defined.
        if not hasattr(cls, "data_url"):
            raise TypeError("Dataset classes must define a 'data_url' attribute.")
        if cls.tags.get("has_ground_truth") and not hasattr(cls, "ground_truth_url"):
            raise TypeError(
                "Dataset classes with 'has_ground_truth' tag True must define a 'ground_truth_url' attribute."
            )
        if cls.tags.get("has_expert_knowledge") and not hasattr(
            cls, "expert_knowledge_url"
        ):
            raise TypeError(
                "Dataset classes with 'has_expert_knowledge' tag True must define an 'expert_knowledge_url' attribute."
            )

        # Step 4: Register the dataset by name and tags.
        name = getattr(cls, "name")
        self._by_name[name] = cls

        raw_tags = getattr(cls, "tags")
        for key, value in raw_tags.items():
            self._by_tag.setdefault((key, value), set()).add(name)

    def list_datasets(self, **tag_filters: Any) -> List[str]:
        """
        List dataset names, optionally filtered by tag key-value pairs.

        Parameters
        ----------
        **tag_filters :
            Tag constraints as keyword arguments. The following tags are supported:
            - has_ground_truth
            - has_expert_knowledge
            - has_missing_data
            - is_simulated
            - is_interventional
            - is_discrete
            - is_continuous
            - is_mixed
            - is_ordinal

        Returns
        -------
        List[str]
            Sorted list of dataset names that satisfy the filters.

        Examples
        --------
        >>> from pgmpy.datasets import DATASET_REGISTRY
        >>> all_datasets = DATASET_REGISTRY.list_datasets()
        >>> discrete_datasets = DATASET_REGISTRY.list_datasets(is_discrete=True)
        ['sachs_discrete']
        >>> mixed_datasets_with_gt = DATASET_REGISTRY.list_datasets(
        ...     is_mixed=True, has_ground_truth=True
        ... )
        ['sachs_mixed']
        """
        # Step 1: If no filters return all.
        if not tag_filters:
            return sorted(self._by_name.keys())

        # Step 2: Gather candidate sets for each tag filter.
        candidate_sets = []

        for key, value in tag_filters.items():
            names_for_tag = self._by_tag.get((key, value), set())
            candidate_sets.append(names_for_tag)

        # Step 3: Intersect candidate sets and return.
        names = candidate_sets[0].copy()
        for s in candidate_sets[1:]:
            names.intersection_update(s)

        return sorted(names)

    def get_dataset(self, name: str) -> Optional[Type["_BaseDataset"]]:
        """
        Get the dataset class by name.

        Parameters
        ----------
        name : str
            Name of the dataset.

        Returns
        -------
        Instance of the Dataset class.
        """
        if name not in self._by_name:
            raise ValueError(f"Dataset '{name}' not found in registry.")
        return self._by_name[name]


DATASET_REGISTRY = _DatasetRegistry()


def register_dataset_class(cls):
    """
    Class decorator to register a dataset class in the DATASET_REGISTRY.

    For example usage see one of the dataset files such as `abalone.py`.
    """
    DATASET_REGISTRY.register(cls)
    return cls


def load_dataset(name: str) -> Dataset:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    Examples
    --------
    >>> from pgmpy.datasets import load_dataset
    >>> data, ground_truth = load_dataset("sachs", load_ground_truth=True)

    """
    dataset_cls = DATASET_REGISTRY.get_dataset(name)
    return Dataset(
        name=name,
        data=dataset_cls.load_dataframe(),
        expert_knowledge=dataset_cls.load_expert_knowledge(),
        ground_truth=dataset_cls.load_ground_truth(),
        tags=dataset_cls.tags,
    )
