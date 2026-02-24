import pandas as pd

from pgmpy.datasets._base import _BaseDataset


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

    base_url = (
        "https://raw.githubusercontent.com/grf-labs/grf/"
        "refs/heads/master/experiments/"
    )

    data_url = base_url + "acic18/synthetic_data.csv"
    ground_truth_url = None
    expert_knowledge_url = None

    missing_values_marker = None

    categorical_variables = ["schoolid", "Z", "C1", "C2", "C3", "XC"]
    ordinal_variables = dict()

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        df = pd.read_csv(cls.data_url)
        # there are some categorical variables
        for col in cls.categorical_variables:
            df[col] = df[col].astype("category")
        return df
