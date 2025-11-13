from __future__ import annotations

import hashlib
import io
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import pandas as pd
import requests

from pgmpy.base import DAG

from .registry import DATASETS, dataset_class

PGMPY_DATA_HOME = os.environ.get(
    "PGMPY_DATA_HOME", os.path.join(Path.home(), ".pgmpy", "data")
)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_df_from_txt(raw: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception as e:
        try:
            return pd.read_csv(io.BytesIO(raw), sep=",", engine="python")
        except Exception:
            raise e


def parse_sachs_graph(
    text_content: str,
) -> Optional[DAG]:  # parses the 'sachs.2005.ground.truth.graph.txt' format
    try:
        lines = [line.strip() for line in text_content.splitlines() if line.strip()]

        nodes_start = -1
        edges_start = -1

        for i, line in enumerate(lines):
            if line.lower().startswith("graph nodes:"):
                nodes_start = i
            elif line.lower().startswith("graph edges:"):
                edges_start = i

        if nodes_start == -1 or edges_start == -1:
            return None

        G = nx.DiGraph()

        node_str = " ".join(lines[nodes_start + 1 : edges_start])
        nodes_list = re.split(r"[;\s]+", node_str)
        for node in nodes_list:
            if node:
                G.add_node(node.strip())

        for line in lines[edges_start + 1 :]:
            match = re.search(r"(\w+)\s*-->\s*(\w+)", line)
            if match:
                try:
                    u = match.group(1).strip()
                    v = match.group(2).strip()

                    if u in G and v in G:
                        G.add_edge(u, v)
                except Exception:
                    continue
        return DAG(G)

    except Exception:
        return None


def parse_tier_graph(
    text_content: str,
) -> Optional[DAG]:  # Parses the 'abalone.knowledge.txt' tiered format.
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


def parse_knowledge_txt(raw: bytes) -> Optional[DAG]:
    text = raw.decode("utf-8-sig", errors="ignore")

    sachs_graph = parse_sachs_graph(text)
    if sachs_graph is not None:
        return sachs_graph

    tier_graph = parse_tier_graph(text)
    if tier_graph is not None:
        return tier_graph
    return None


class BaseDataset:
    name: str = ""
    description: str = ""
    tags: Dict[str, Any] = {}

    @classmethod
    def cache_path(cls, key: str) -> str:
        ensure_dir(PGMPY_DATA_HOME)
        safe = hashlib.sha256(f"{cls.name}:{key}".encode()).hexdigest()[:16]
        ext = ".bin"
        if key.startswith("data"):
            ext = ".parquet"
        elif key.startswith("ground_truth"):
            ext = ".json"
        return os.path.join(
            PGMPY_DATA_HOME, f"{cls.name}_{key.replace(':','-')}_{safe}{ext}"
        )

    @classmethod
    def _fetch(cls, key: str) -> bytes:
        raise NotImplementedError

    @classmethod
    def load_or_fetch(cls, key: str):
        cache_path = cls.cache_path(key)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                raw = f.read()
        else:
            raw = cls._fetch(key)
            with open(cache_path, "wb") as f:
                f.write(raw)
        return raw

    @classmethod
    def load(cls):
        raise NotImplementedError

    @classmethod
    def load_ground_truth(cls):
        return None


def load_dataset(
    name: str, load_ground_truth: bool = True
) -> Tuple[pd.DataFrame, Optional[DAG]]:
    ds = DATASETS.get(name)
    if ds is None:
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {DATASETS.list_all()}"
        )

    try:
        X = ds.load()
    except TypeError:
        X = ds.load()

    gt = ds.load_ground_truth() if load_ground_truth else None

    if gt is None:
        gt = DAG()

    data_cols = set(X.columns)
    gt_nodes = set(gt.nodes())

    if data_cols != gt_nodes:

        missing_in_gt = data_cols - gt_nodes
        gt.add_nodes_from(missing_in_gt)

        missing_in_data = gt_nodes - data_cols
        if missing_in_data:
            import warnings

            warnings.warn(
                f"Ground truth for '{name}' has nodes not in data: {missing_in_data}"
            )

    return X, gt


@dataset_class
class Abalone(BaseDataset):
    name = "abalone"
    tags = {
        "has_ground_truth": True,
        "is_simulated": False,
        "n_variables": 9,
        "n_samples": 4177,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/abalone/"

    VARIANT_URLS = {
        "continuous": base_url + "data/abalone.continuous.txt",
        "mixed_numeric": base_url + "data/abalone.mixed.numeric.txt",
        "mixed_max3": base_url + "data/abalone.mixed.maximum.3.txt",
    }
    GROUND_TRUTH_URL = base_url + "ground.truth/abalone.knowledge.txt"
    DEFAULT_VARIANT = "mixed_numeric"

    @classmethod
    def _fetch(cls, key: str) -> bytes:
        if key.startswith("data:"):
            variant = key.split(":", 1)[1]
            url = cls.VARIANT_URLS.get(variant)
            if url is None:
                raise KeyError(
                    f"Unknown abalone variant '{variant}'. Available: {list(cls.VARIANT_URLS.keys())}"
                )
        elif key == "ground_truth":
            url = cls.GROUND_TRUTH_URL
        else:
            raise KeyError(f"Unknown fetch key '{key}'.")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content

    @classmethod
    def load(cls, variant: Optional[str] = None) -> pd.DataFrame:
        v = variant or cls.DEFAULT_VARIANT
        raw = cls.load_or_fetch(f"data:{v}")
        return load_df_from_txt(raw)

    @classmethod
    def load_ground_truth(cls) -> Optional[DAG]:
        raw = cls.load_or_fetch("ground_truth")
        return parse_knowledge_txt(raw)


def load_abalone(
    variant: Optional[str] = None, load_ground_truth: bool = True
) -> Tuple[pd.DataFrame, Optional[DAG]]:
    df = Abalone.load(variant=variant)
    gt = Abalone.load_ground_truth() if load_ground_truth else None
    return df, gt


@dataset_class
class Sachs(BaseDataset):
    name = "sachs"
    tags = {
        "has_ground_truth": True,
        "is_simulated": False,
        "n_variables": 11,
        "n_samples": 7466,
        "is_discrete": True,
        "is_continuous": True,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/refs/heads/main/real/sachs/"

    VARIANT_URLS = {
        "continuous": base_url + "data/sachs.2005.continuous.txt",
        "discrete": base_url + "data/sachs.2005.discrete.txt",
        "logxplus10_continuous": base_url + "data/sachs.2005.logxplus10.continuous.txt",
        "jittered_experimental": base_url
        + "data/sachs.2005.with.jittered.experimental.continuous.txt",
        "logxplus10_jittered_experimental": base_url
        + "data/sachs.2005.logxplus10.jittered.eperimental.continuous.txt",
        "mixed_maximum_2": base_url
        + "data/sachs.2005.continuous.discrete.experimental.mixed.maximum.2.txt",
    }

    GROUND_TRUTH_URL = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    DEFAULT_VARIANT = "continuous"

    @classmethod
    def _fetch(cls, key: str) -> bytes:
        if key.startswith("data:"):
            variant = key.split(":", 1)[1]
            url = cls.VARIANT_URLS.get(variant)
            if url is None:
                raise KeyError(
                    f"Unknown sachs variant '{variant}'. Available: {list(cls.VARIANT_URLS.keys())}"
                )
        elif key == "ground_truth":
            url = cls.GROUND_TRUTH_URL
        else:
            raise KeyError(f"Unknown fetch key '{key}'.")

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content

    @classmethod
    def load(cls, variant: Optional[str] = None) -> pd.DataFrame:
        v = variant or cls.DEFAULT_VARIANT
        raw = cls.load_or_fetch(f"data:{v}")
        return load_df_from_txt(raw)

    @classmethod
    def load_ground_truth(cls) -> Optional[DAG]:
        raw = cls.load_or_fetch("ground_truth")
        return parse_knowledge_txt(raw)


def load_sachs(
    variant: Optional[str] = None, load_ground_truth: bool = True
) -> Tuple[pd.DataFrame, Optional[DAG]]:
    df = Sachs.load(variant=variant)
    gt = Sachs.load_ground_truth() if load_ground_truth else None
    return df, gt
