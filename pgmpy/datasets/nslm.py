import pandas
import io

from pgmpy.base import DAG
from pgmpy.datasets._base import _BaseDataset
from pgmpy.estimators import ExpertKnowledge


class NSLM(_BaseDataset):

    _tags = {
        "name": "nslm",
        "n_variables": 13,
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

    categorical_variables = ["C1", "C2", "C3", "XC", "schoolid", "Z"]
    ordinal_variables = dict()


    @classmethod
    def load_dataframe(cls) -> pandas.DataFrame:
        raw_data = cls._get_raw_data("data", cls.data_url)

        dataframe = pandas.read_csv(io.BytesIO(raw_data), sep=",")
        return dataframe