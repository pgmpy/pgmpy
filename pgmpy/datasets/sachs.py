import re
from typing import Optional

import networkx as nx

from pgmpy.base import DAG
from pgmpy.datasets import dataset_class
from pgmpy.datasets._base import _BaseDataset


@dataset_class
class Sachs(_BaseDataset):
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

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

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

    @staticmethod
    def _parse_sachs_graph(text_content: str) -> Optional[DAG]:
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

    @classmethod
    def _parse_ground_truth(cls, raw: bytes) -> Optional[DAG]:
        text = raw.decode("utf-8-sig", errors="ignore")
        return cls._parse_sachs_graph(text)
