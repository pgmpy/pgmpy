from __future__ import annotations

import hashlib
import io
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import networkx as nx
import pandas as pd

from pgmpy.base import DAG
from pgmpy.global_vars import PGMPY_DATA_HOME, logger
from pgmpy.utils._safe_import import _safe_import

requests = _safe_import("requests")


class _BaseDataset:
    @staticmethod
    def _ensure_dir(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_df_from_txt(raw: bytes) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(raw), sep="\t")

    @classmethod
    def cache_path(cls, key: str) -> str:
        cls._ensure_dir(PGMPY_DATA_HOME)
        safe = hashlib.sha256(f"{cls.name}:{key}".encode()).hexdigest()[:16]

        if key.startswith("data"):
            ext = ".csv"
        elif key.startswith("ground_truth"):
            ext = ".txt"
        else:
            raise ValueError(
                f"Unknown key type: {key}. Must start with 'data' or 'ground_truth'."
            )
        return os.path.join(
            PGMPY_DATA_HOME, f"{cls.name}_{key.replace(':','-')}_{safe}{ext}"
        )

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
        v = variant or cls.DEFAULT_VARIANT
        if v not in cls.VARIANT_URLS:
            raise KeyError(
                f"Unknown variant '{v}'. Available: {list(cls.VARIANT_URLS.keys())}"
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


class DatasetRegistry:
    def __init__(self) -> None:
        self._by_name = {}
        self._by_tag = {}

    def register(self, cls):
        self._by_name[cls.name] = cls
        for tag in cls.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = set()
            self._by_tag[tag].add(cls.name)

    def list_all(self, tag=None):
        if tag is None:
            return list(self._by_name.keys())
        else:
            return list(self._by_tag.get(tag, []))

    def get(self, name):
        return self._by_name.get(name, None)


DATASETS = DatasetRegistry()


def dataset_class(cls):
    DATASETS.register(cls)
    return cls


def load_dataset(
    name: str, variant: Optional[str] = None, load_ground_truth: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[DAG]]]:

    dataset = DATASETS.get(name)
    if dataset is None:
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {DATASETS.list_all()}"
        )
    X = dataset.load(variant=variant)

    if not load_ground_truth:
        return X

    ground_truth = dataset.load_ground_truth()

    if ground_truth is None:
        ground_truth = DAG()

    data_cols = set(X.columns)
    gt_nodes = set(ground_truth.nodes())

    if data_cols != gt_nodes:
        missing_in_gt = data_cols - gt_nodes
        ground_truth.add_nodes_from(missing_in_gt)

        missing_in_data = gt_nodes - data_cols
        if missing_in_data:
            logger.warning(
                f"Ground truth for '{name}' has nodes not in data: {missing_in_data}"
            )

    return X, ground_truth
