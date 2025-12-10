import re
from typing import Optional

import networkx as nx

from pgmpy.base import DAG
from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


class _BaseSachs:
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


@register_dataset_class
class SachsMixed(_BaseDataset, _BaseSachs):
    name = "sachs_mixed"
    tags = {
        "n_variables": 20,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = (
        base_url
        + "data/sachs.2005.continuous.discrete.experimental.mixed.maximum.2.txt"
    )
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"


@register_dataset_class
class SachsContinuous(_BaseDataset, _BaseSachs):
    name = "sachs_continuous"
    tags = {
        "n_variables": 11,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.continuous.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"


@register_dataset_class
class SachsDiscrete(_BaseDataset, _BaseSachs):
    name = "sachs_discrete"
    tags = {
        "n_variables": 11,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": True,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.discrete.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"


@register_dataset_class
class SachsContinuousLogScale(_BaseDataset, _BaseSachs):
    name = "sachs_continuous_logscale"
    tags = {
        "n_variables": 11,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.logxplus10.continuous.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"


@register_dataset_class
class SachsContinuousJitteredLogScale(_BaseDataset, _BaseSachs):
    name = "sachs_continuous_jittered_logscale"
    tags = {
        "n_variables": 20,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = (
        base_url + "data/sachs.2005.logxplus10.jittered.experimental.continuous.txt"
    )
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"


@register_dataset_class
class SachsContinuousJittered(_BaseDataset, _BaseSachs):
    name = "sachs_continuous_jittered"
    tags = {
        "n_variables": 20,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.with.jittered.experimental.continuous.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"
