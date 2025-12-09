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
    @classmethod
    def load(cls, variant: Optional[str] = None) -> pd.DataFrame:
        # Step 0: Determine variant to load and get the URL.
        if variant is None:
            variant = cls.default_variant

        if variant not in cls.variant_urls:
            raise ValueError(
                f"Unknown variant '{variant}'. Available options are: {list(cls.variant_urls.keys())}"
            )

        data_url = cls.variant_urls[variant]
        has_ground_truth = cls.tags.get("has_ground_truth")

        # Step 1: Create cache path and load or fetch the data.
        cache_dir_path = os.path.join(
            PGMPY_DATA_HOME,
            hashlib.sha256(f"{cls.name}_{variant}_url".encode()).hexdigest(),
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

            resp = requests.get(data_url, timeout=60)
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
            return dataset.load(variant)
    else:
        return dataset.load(variant)[0]
