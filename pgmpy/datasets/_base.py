from __future__ import annotations

import hashlib
import io
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import networkx as nx
import pandas as pd

from pgmpy.base import DAG
from pgmpy.global_vars import PGMPY_DATA_HOME
from pgmpy.utils._safe_import import _safe_import

requests = _safe_import("requests")


class _BaseDataset:
    @staticmethod
    def _load_df_from_txt(raw: bytes) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(raw), sep="\t")

    @classmethod
    def cache_path(cls, key: str) -> str:
        Path(path).mkdir(parents=True, exist_ok=True)
        return os.path.join(
            PGMPY_DATA_HOME, f"{cls.name}_{key.replace(':','-')}_{safe}{ext}"
        )

        # safe = hashlib.sha256(f"{cls.name}:{key}".encode()).hexdigest()[:16]

        # if key.startswith("data"):
        #     ext = ".csv"
        # elif key.startswith("ground_truth"):
        #     ext = ".txt"
        # else:
        #     raise ValueError(
        #         f"Unknown key type: {key}. Must start with 'data' or 'ground_truth'."
        #     )

    @classmethod
    def load_or_fetch(cls, key: str, url: str) -> bytes:
        cache_path = cls.cache_path(key)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                raw = f.read()
        else:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            raw = resp.content
            with open(cache_path, "wb") as f:
                f.write(raw)
        return raw

    @classmethod
    def load(cls, variant: Optional[str] = None) -> pd.DataFrame:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        if variant not in cls.VARIANT_URLS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available options are: {list(cls.VARIANT_URLS.keys())}"
            )

        url = cls.VARIANT_URLS[v]
        raw = cls.load_or_fetch(f"data:{v}", url)
        return cls._load_df_from_txt(raw)

    @classmethod
    def load_ground_truth(cls) -> Optional[DAG]:
        if not getattr(cls, "GROUND_TRUTH_URL", None):
            return None

        raw = cls.load_or_fetch("ground_truth", cls.GROUND_TRUTH_URL)
        return cls._parse_ground_truth(raw)

    @staticmethod
    def _parse_tier_graph(text_content: str) -> Optional[DAG]:
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
    def _parse_ground_truth(cls, raw: bytes) -> Optional[DAG]:
        text = raw.decode("utf-8-sig", errors="ignore")
        return cls._parse_tier_graph(text)


class _DatasetRegistry:
    """
    Registry for dataset classes.

    Example
    -------
    >>> from pgmpy.datasets import DATASETS
    >>> all_datasets = DATASETS.list_datasets()
    >>> filtered_datasets = DATASETS.list_datasets(has_ground_truth=True, is_mixed=True)

    """

    _REQUIRED_TAGS = {
        "has_ground_truth",
        "is_simulated",
        "n_variables",
        "n_samples",
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

        # Step 3: Register the dataset by name and tags.
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


DATASETS = _DatasetRegistry()


def dataset_class(cls):
    """
    Class decorator to register a dataset class in the DATASETS registry.

    For example usage see one of the dataset files such as `abalone.py`.
    """
    DATASETS.register(cls)
    return cls


def load_dataset(
    name: str, variant: Optional[str] = None, load_ground_truth: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[DAG]]]:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    variant : str, default=None
        Variant of the dataset to load. If None, the dataset's default variant
        is loaded.

    load_ground_truth : bool, default=True
        Whether to load the ground truth DAG along with the data.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[DAG]]]
        If `load_ground_truth` is False, returns only the data as a DataFrame.
        If True, returns a tuple of (data, ground_truth), where ground_truth
        is a DAG or None if not available.

    Examples
    --------
    >>> from pgmpy.datasets import load_dataset
    >>> data, ground_truth = load_dataset("sachs", load_ground_truth=True)

    """

    dataset = DATASETS.get_dataset(name)

    if load_ground_truth:
        if not dataset.tags.get("has_ground_truth"):
            raise ValueError(f"Dataset '{name}' does not have ground truth available.")
        else:
            return (dataset.load(variant), dataset.load_ground_truth())
    else:
        return dataset.load(variant)
