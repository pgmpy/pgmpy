import io

import pandas as pd

from pgmpy.datasets._base import _BaseDataset


class IHDP(_BaseDataset):
    """
    References
    ----------
    .. [1] Hill, J. (2011). Bayesian Nonparametric Modeling for Causal Inference.
           Journal of Computational and Graphical Statistics, 20(1), 217-240.
    .. [2] https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP
    """

    _tags = {
        "name": "ihdp",
        "n_variables": 30,
        "n_samples": 747,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/"
        "IHDP/csv/"
    )
    data_url = base_url + "ihdp_npci_1.csv"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = ["treatment"]
    ordinal_variables = dict()

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        raw_data = cls._get_raw_data("data", cls.data_url)
        column_names = [
            "treatment",
            "y_factual",
            "y_cfactual",
            "mu0",
            "mu1",
            *[f"x{i}" for i in range(1, 26)],
        ]

        df = pd.read_csv(io.BytesIO(raw_data), sep=",", header=None, names=column_names)
        df["treatment"] = df["treatment"].astype("category")
        return df
