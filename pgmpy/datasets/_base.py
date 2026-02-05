from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects
from skbase.utils.dependencies import _safe_import

from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge
from pgmpy.global_vars import PGMPY_DATA_HOME

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


class _BaseDataset(BaseObject):
    """
    Base class for all datasets in pgmpy.
    Inherits from skbase.base.BaseObject to utilize its tag and lookup functionality.
    """

    # define tags
    _tags = {
        "name": None,
        "n_variables": None,
        "n_samples": None,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }

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
        name = cls.get_class_tag("name")
        cache_dir_path = os.path.join(
            PGMPY_DATA_HOME,
            hashlib.sha256(f"{name}_{cls.base_url}".encode()).hexdigest(),
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
        if cls.get_class_tag("has_missing_data"):
            df.replace(cls.missing_values_marker, pd.NA, inplace=True)
        if cls.get_class_tag("has_index_col"):
            df.drop(df.columns[0], axis=1, inplace=True)
        if len(cls.categorical_variables) > 0:
            for col in cls.categorical_variables:
                df[col] = df[col].astype("category")
        if len(cls.ordinal_variables) > 0:
            for col, order in cls.ordinal_variables.items():
                cat_type = pd.CategoricalDtype(categories=order, ordered=True)
                df[col] = df[col].astype(cat_type)
        return df

    @classmethod
    def load_expert_knowledge(cls) -> ExpertKnowledge:
        """Fetches/reads from cache the expert knowledge associated with the dataset."""
        if not cls.get_class_tag("has_expert_knowledge"):
            return None

        raw_data = cls._get_raw_data("expert_knowledge", cls.expert_knowledge_url)
        expert_knowledge = cls._parse_expert_knowledge(raw_data)
        return expert_knowledge

    @classmethod
    def load_ground_truth(cls) -> DAG:
        """Fetches/reads from cache the ground truth DAG associated with the dataset."""
        if not cls.get_class_tag("has_ground_truth"):
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
            shutil.rmtree(PGMPY_DATA_HOME)


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
        mean = [0] * cls.get_class_tag("n_variables")
        data = pd.DataFrame(
            np.random.multivariate_normal(
                mean, cov_matrix.values, size=cls.get_class_tag("n_samples")
            ),
            columns=cov_matrix.columns,
        )
        return data


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
    >>> dataset = load_dataset("sachs_mixed")
    >>> df = dataset.data
    >>> ground_truth = dataset.ground_truth
    """
    all_datasets = all_objects(
        object_types=_BaseDataset, package_name="pgmpy.datasets", return_names=False
    )

    target_cls = None
    for cls in all_datasets:
        if cls.get_class_tag("name") == name:
            target_cls = cls
            break
    if target_cls is None:
        raise ValueError(
            f"Dataset with name '{name}' not found. Please use list_datasets() to see available datasets."
        )

    return Dataset(
        name=name,
        data=target_cls.load_dataframe(),
        expert_knowledge=target_cls.load_expert_knowledge(),
        ground_truth=target_cls.load_ground_truth(),
        tags=target_cls.get_class_tags(),
    )


def list_datasets(**filter_tags) -> list[str]:
    """
    Returns a list of all available datasets, optionally filtered by a query string.

    Parameters
    ----------
    **filter_tags : optional arguments
        If specified, returns only datasets matching the provided tag filters. Any dataset tag can be used as a filter.
        Available tags:
            - n_variables
            - n_samples
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
    list of str
        A sorted list of available dataset names.

    Examples
    --------
    >>> from pgmpy.datasets import list_datasets
    >>> list_datasets()
    ['abalone_continuous', 'abalone_mixed', ..., 'sachs_continuous', ...]

    >>> list_datasets(is_discrete=True, has_ground_truth=True)
    ['sachs_discrete']
    """
    all_datasets = all_objects(
        object_types=_BaseDataset,
        package_name="pgmpy.datasets",
        return_names=False,
        filter_tags=filter_tags,
    )

    dataset_names = [
        cls.get_class_tag("name")
        for cls in all_datasets
        if cls.get_class_tag("name") is not None
    ]

    return sorted(dataset_names)
