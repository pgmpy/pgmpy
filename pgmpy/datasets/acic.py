import io
from urllib.error import HTTPError, URLError

import pandas as pd

from pgmpy.datasets._base import _BaseDataset


class ACIC2016(_BaseDataset):
    """
    Atlantic Causal Inference Conference (ACIC) 2016 Benchmark Dataset.

    Semi-synthetic dataset for evaluating causal inference methods. Covariates
    (x1-x58) are from the real Collaborative Perinatal Project (CPP) study.
    Treatment (z) and outcome (y) are simulated under a known mechanism,
    providing ground-truth potential outcomes (mu0, mu1) for benchmarking.

    This loader provides simulation #1 from 1600 available simulations.

    Variables:
        - x1-x58: Covariates from the CPP study (mix of integer and nominal types)
        - x2, x21, x24: Categorical (nominal, letter-coded A/B/C/...)
        - z: Binary treatment (0=control, 1=treated)
        - y: Continuous observed outcome
        - mu0: True potential outcome under control
        - mu1: True potential outcome under treatment

    True ATE = ``(dataset.data['mu1'] - dataset.data['mu0']).mean()``

    References
    ----------
    .. [1] Dorie, V., Hill, J., Shalit, U., Scott, M., & Cervone, D. (2019).
           Automated versus do-it-yourself methods for causal inference:
           Lessons learned from a data analysis competition. Statistical Science,
           34(1), 43-68.
    .. [2] IBM Research. causallib. https://github.com/BiomedSciAI/causallib
    """

    _tags = {
        "name": "acic_2016",
        "n_variables": 62,
        "n_samples": 4802,
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
        "https://raw.githubusercontent.com/pgmpy/example_datasets/"
        "refs/heads/main/simulated/acic-2016/"
    )

    # Canonical source path. If unavailable, loader falls back to source components.
    data_url = base_url + "data/acic.2016.sim.1.mixed.txt"
    fallback_base_url = (
        "https://raw.githubusercontent.com/BiomedSciAI/causallib/"
        "master/causallib/datasets/data/acic_challenge_2016/"
    )
    fallback_x_url = fallback_base_url + "x.csv"
    fallback_outcomes_url = fallback_base_url + "zymu_1.csv"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = ["x2", "x21", "x24"]
    ordinal_variables = dict()

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        """Fetches/reads from cache ACIC simulation data."""
        try:
            raw_data = cls._get_raw_data("data", cls.data_url)
            data = pd.read_csv(io.BytesIO(raw_data), sep="\t")

            for col in cls.categorical_variables:
                data[col] = data[col].astype("category")

            return data
        except (HTTPError, URLError):
            raw_x = cls._get_raw_data("fallback_x", cls.fallback_x_url)
            raw_outcomes = cls._get_raw_data(
                "fallback_outcomes", cls.fallback_outcomes_url
            )

            x = pd.read_csv(io.BytesIO(raw_x))
            outcomes = pd.read_csv(io.BytesIO(raw_outcomes))

            x.columns = [col.replace("x_", "x") for col in x.columns]
            data = pd.concat([x, outcomes[["z", "mu0", "mu1"]]], axis=1)
            data["y"] = (
                outcomes["y0"] * (1 - outcomes["z"]) + outcomes["y1"] * outcomes["z"]
            )

            for col in cls.categorical_variables:
                data[col] = data[col].astype("category")

            return data[[*[f"x{i}" for i in range(1, 59)], "z", "y", "mu0", "mu1"]]
