from __future__ import annotations

import hashlib
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import networkx as nx
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


class _BaseDataset:
    @staticmethod
    def _parse_tier_graph(raw_ground_truth: str) -> Optional[DAG]:
        text_content = raw_ground_truth.decode("utf-8-sig", errors="ignore")

        try:
            lines = [line.strip() for line in text_content.splitlines() if line.strip()]
            start_index = -1
            for i, line in enumerate(lines):
                if line.lower() == "addtemporal":
                    start_index = i
                    break

            if start_index == -1:
                return None

            G = nx.DiGraph()
            tiers = []

            for line in lines[start_index + 1 :]:
                match = re.match(r"^\d+\s+(.*)", line)
                if match:
                    vars_list = match.group(1).split()
                    if vars_list:
                        tiers.append(vars_list)
                        G.add_nodes_from(vars_list)
                elif line.startswith("/") or "direct" in line.lower():
                    break

            for i in range(len(tiers)):
                for j in range(i + 1, len(tiers)):
                    for u in tiers[i]:
                        for v in tiers[j]:
                            if u in G and v in G and u != v:
                                G.add_edge(u, v)
            return DAG(G)
        except Exception:
            return None

    @classmethod
    def _get_raw_data(cls, data_type, url) -> bytes:
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
    def load_data(cls) -> pd.DataFrame:
        raw_data = cls._get_raw_data("data", cls.data_url)
        df = pd.read_csv(io.BytesIO(raw_data), sep="\t")
        return df

    @classmethod
    def load_expert_knowledge(cls) -> ExpertKnowledge:
        if not cls.tags.get("has_expert_knowledge"):
            return None

        raw_data = cls._get_raw_data("expert_knowledge", cls.expert_knowledge_url)
        return raw_data

        # TODO: Construct an ExpertKnowledge object from raw_data

    @classmethod
    def load_ground_truth(cls) -> DAG:
        if not cls.tags.get("has_ground_truth"):
            return None

        raw_data = cls._get_raw_data("ground_truth", cls.ground_truth_url)

        return cls._parse_ground_truth(raw_data)

    @classmethod
    def load(cls) -> pd.DataFrame:

        has_ground_truth = cls.tags.get("has_ground_truth")

        # Step 1: Create cache path and load or fetch the data.
        cache_dir_path = os.path.join(
            PGMPY_DATA_HOME,
            hashlib.sha256(f"{cls.name}_url".encode()).hexdigest(),
        )
        cache_data_path = os.path.join(cache_dir_path, "data")
        cache_ground_truth_path = os.path.join(cache_dir_path, "ground_truth")

        if os.path.exists(cache_dir_path):
            with open(cache_data_path, "rb") as f:
                raw_data = f.read()
            if has_ground_truth:
                with open(cache_ground_truth_path, "rb") as f:
                    raw_ground_truth = f.read()

        else:
            os.makedirs(cache_dir_path, exist_ok=True)

            resp = requests.get(cls.data_url, timeout=60)
            resp.raise_for_status()
            raw_data = resp.content
            with open(cache_data_path, "wb") as f:
                f.write(raw_data)

            if has_ground_truth:
                resp_gt = requests.get(cls.ground_truth_url, timeout=60)
                resp_gt.raise_for_status()
                raw_ground_truth = resp_gt.content
                with open(cache_ground_truth_path, "wb") as f:
                    f.write(raw_ground_truth)

        # Step 2: Parse and return the data.
        df = pd.read_csv(io.BytesIO(raw_data), sep="\t")
        if has_ground_truth:
            ground_truth = cls._parse_ground_truth(raw_ground_truth)
            return df, ground_truth
        else:
            return df, None

    @staticmethod
    def clear_cache() -> None:
        """
        Clears the cached data for all datasets.
        """
        if os.path.exists(PGMPY_DATA_HOME):
            Path.rmdir(PGMPY_DATA_HOME)


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
            Tag constraints as keyword arguments. A dataset is included if
            for every (key, value) in tag_filters, its tags dict satisfies
            tags[key] == value.

        Returns
        -------
        List[str]
            Sorted list of dataset names that satisfy the filters.
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


def load_dataset(name: str) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[DAG]]]:
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
        data=dataset_cls.load_data(),
        expert_knowledge=dataset_cls.load_expert_knowledge(),
        ground_truth=dataset_cls.load_ground_truth(),
        tags=dataset_cls.tags,
    )
