# This extension template provides instructions to add new datasets to pgmpy.
#
# Please follow the following steps:

# 2. Go through the file and address all the TODOs.


import pandas

from pgmpy.base import DAG
from pgmpy.datasets._base import _BaseDataset
from pgmpy.estimators import ExpertKnowledge


class NSLM(_BaseDataset):

    _tags = {
        "name": "nslm",
        "n_variables": 11,
        "n_samples": 10391,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/grf-labs/grf/refs/heads/master/experiments/"
    data_url = base_url + "acic18/synthetic_data.csv"
    ground_truth_url = None
    expert_knowledge_url = None

    missing_values_marker = None

    categorical_variables = ["C1", "C2", "C3", "XC"]
    ordinal_variables = dict()

    # TODO: If the ground truth file is in dagitty format, remove the following `load_ground_truth` method.
    @classmethod
    def load_ground_truth(cls) -> DAG:
        if not cls.get_class_tag("has_ground_truth"):
            return None

        _ = cls._get_raw_data("ground_truth", cls.ground_truth_url).decode(
            "utf-8-sig", errors="ignore"
        )
        # TODO: Add logic for parsing the data from the line above into a `pgmpy.base.DAG` object.
        dag = None
        return dag

    # TODO: If the data is in tabular text format, remove the following `load_dataframe` method.
    @classmethod
    def load_dataframe(cls) -> pandas.DataFrame:
        raw_data = cls._get_raw_data("data", cls.data_url)

        # TODO: Add logic to construct a pandas DataFrame object from data in line above.
        dataframe = pandas.read_csv(io.BytesIO(raw_data), sep=",")
        return dataframe

    # TODO: If the expert knowledge is in the expected format, remove the following `load_expert_knowledge` method.
    @classmethod
    def load_expert_knowledge(cls) -> ExpertKnowledge:
        if not cls.get_class_tag("has_expert_knowledge"):
            return None

        _ = cls._get_raw_data("expert_knowledge", cls.expert_knowledge_url)

        # TODO: Add logic to construct a `pgmpy.estimator.ExpertKnowledge` object from data in line above.
        expert_knowledge = None
        return expert_knowledge
