from __future__ import annotations

import io
import re
import warnings

import pandas as pd

from pgmpy.base import DAG
from pgmpy.datasets._base import BaseDataset


class BaseFeedbacksDataset(BaseDataset):
    """
    References
    ----------
    - :footcite:t:`sanchezromero_2019`
    """

    _tags = {
        "is_simulated": True,
        "has_ground_truth": True,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
        "n_samples": 500,
    }

    base_url = "simulated/feedbacks"

    network_name: str | None = None
    n_simulations = 60

    @classmethod
    def load_dataframe(cls, sim_id: int = 1, n_samples=None, seed=None) -> pd.DataFrame:
        if not (1 <= sim_id <= cls.n_simulations):
            raise ValueError(f"sim_id must be between 1 and {cls.n_simulations}. Got {sim_id}.")

        filename = f"data/{cls.network_name}/sim-{sim_id:02d}.{cls.network_name}.continuous.txt"
        raw_data = cls._get_raw_data(filename)
        df = pd.read_csv(io.BytesIO(raw_data), sep="\t")

        if n_samples is not None:
            if n_samples > len(df):
                warnings.warn(
                    f"Requested {n_samples} samples but dataset only has {len(df)}. Returning all {len(df)} rows."
                )
            else:
                df = df.iloc[:n_samples].reset_index(drop=True)

        return df

    @classmethod
    def load_ground_truth(cls, **kwargs) -> DAG:
        filename = f"ground.truth/{cls.network_name}/{cls.network_name}.ground.truth.graph.txt"
        raw_data = cls._get_raw_data(filename).decode("utf-8-sig", errors="ignore")
        return cls._parse_tetrad_graph(raw_data)

    @staticmethod
    def _parse_tetrad_graph(text: str) -> DAG:
        lines = [line.strip() for line in text.strip().splitlines()]

        nodes_idx = lines.index("Graph Nodes:")
        edges_idx = lines.index("Graph Edges:")

        nodes = [node for node in lines[nodes_idx + 1].split(",") if node]

        graph = DAG()
        graph.add_nodes_from(nodes)

        edge_pattern = re.compile(r"^\d+\.\s+(\S+)\s+-->\s+(\S+)")
        for line in lines[edges_idx + 1 :]:
            if not line:
                continue
            match = edge_pattern.match(line)
            if match is None:
                continue
            source, target = match.groups()
            graph.add_edge(source, target)

        return graph


class FeedbacksNetwork1Amp(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network1_amp", "n_variables": 5}
    network_name = "Network1_amp"


class FeedbacksNetwork2Amp(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network2_amp", "n_variables": 5}
    network_name = "Network2_amp"


class FeedbacksNetwork3Amp(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network3_amp", "n_variables": 5}
    network_name = "Network3_amp"


class FeedbacksNetwork4Amp(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network4_amp", "n_variables": 10}
    network_name = "Network4_amp"


class FeedbacksNetwork5Amp(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network5_amp", "n_variables": 5}
    network_name = "Network5_amp"


class FeedbacksNetwork5Cont(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network5_cont", "n_variables": 5}
    network_name = "Network5_cont"


class FeedbacksNetwork5ContP3N7(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network5_cont_p3n7", "n_variables": 5}
    network_name = "Network5_cont_p3n7"


class FeedbacksNetwork5ContP7N3(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network5_cont_p7n3", "n_variables": 5}
    network_name = "Network5_cont_p7n3"


class FeedbacksNetwork6Amp(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network6_amp", "n_variables": 8}
    network_name = "Network6_amp"


class FeedbacksNetwork6Cont(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network6_cont", "n_variables": 8}
    network_name = "Network6_cont"


class FeedbacksNetwork7Amp(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network7_amp", "n_variables": 6}
    network_name = "Network7_amp"


class FeedbacksNetwork7Cont(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network7_cont", "n_variables": 6}
    network_name = "Network7_cont"


class FeedbacksNetwork8AmpAmp(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network8_amp_amp", "n_variables": 8}
    network_name = "Network8_amp_amp"


class FeedbacksNetwork8AmpCont(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network8_amp_cont", "n_variables": 8}
    network_name = "Network8_amp_cont"


class FeedbacksNetwork8ContAmp(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network8_cont_amp", "n_variables": 8}
    network_name = "Network8_cont_amp"


class FeedbacksNetwork9AmpAmp(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network9_amp_amp", "n_variables": 9}
    network_name = "Network9_amp_amp"


class FeedbacksNetwork9AmpCont(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network9_amp_cont", "n_variables": 9}
    network_name = "Network9_amp_cont"


class FeedbacksNetwork9ContAmp(BaseFeedbacksDataset):
    _tags = {"name": "feedbacks_network9_cont_amp", "n_variables": 9}
    network_name = "Network9_cont_amp"
