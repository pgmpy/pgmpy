import io

import pandas as pd

from pgmpy.datasets._base import _BaseDataset


class _JobsMixin:
    _common_tags = {
        "n_variables": 10,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    columns = [
        "training",
        "age",
        "education",
        "black",
        "hispanic",
        "married",
        "no_degree",
        "re74",
        "re75",
        "re78",
    ]

    categorical_variables = [
        "training",
        "black",
        "hispanic",
        "married",
        "no_degree",
    ]

    ordinal_variables = dict()


class JobsLalonde(_JobsMixin, _BaseDataset):
    """
    References
    ----------
    .. [1] https://users.nber.org/~rdehejia/nswdata2.html
    .. [2] Rajeev Dehejia and Sadek Wahba, "Causal Effects in Non-Experimental Studies:
    Reevaluating the Evaluation of Training Programs," Journal of the American Statistical
    Association, Vol. 94, No. 448 (December 1999), pp. 1053-1062.
    .. [3] Rajeev Dehejia and Sadek Wahba, "Propensity Score Matching Methods for Non-
    Experimental Causal Studies," Review of Economics and Statistics, Vol. 84, (February
    2002), pp. 151-161.
    .. [4] Robert Lalonde, "Evaluating the Econometric Evaluations of Training Programs," American
    Economic Review, Vol. 76 (1986), pp. 604-620.
    """

    _tags = {
        **_JobsMixin._common_tags,
        "name": "jobs_lalonde",
        "n_samples": 445,
        "is_interventional": True,
    }

    base_url = "https://users.nber.org/~rdehejia/data/"
    data_url_treated = base_url + "nswre74_treated.txt"
    data_url_control = base_url + "nswre74_control.txt"

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        """Fetches/reads from cache the data associated with the dataset."""
        raw_target = cls._get_raw_data("data_target", cls.data_url_treated)
        raw_control = cls._get_raw_data("data_control", cls.data_url_control)
        df_treated = pd.read_csv(
            io.BytesIO(raw_target),
            sep=r"\s+",
            header=None,
            names=cls.columns,
        )
        df_control = pd.read_csv(
            io.BytesIO(raw_control),
            sep=r"\s+",
            header=None,
            names=cls.columns,
        )
        df = pd.concat([df_treated, df_control], ignore_index=True)
        for col in cls.categorical_variables:
            df[col] = df[col].astype("category")
        return df


class JobsPSID(_JobsMixin, _BaseDataset):
    """
    References
    ----------
    .. [1] https://users.nber.org/~rdehejia/nswdata2.html
    .. [2] Rajeev Dehejia and Sadek Wahba, "Causal Effects in Non-Experimental Studies:
    Reevaluating the Evaluation of Training Programs," Journal of the American Statistical
    Association, Vol. 94, No. 448 (December 1999), pp. 1053-1062.
    .. [3] Rajeev Dehejia and Sadek Wahba, "Propensity Score Matching Methods for Non-
    Experimental Causal Studies," Review of Economics and Statistics, Vol. 84, (February
    2002), pp. 151-161.
    .. [4] Robert Lalonde, "Evaluating the Econometric Evaluations of Training Programs," American
    Economic Review, Vol. 76 (1986), pp. 604-620.
    """

    _tags = {
        **_JobsMixin._common_tags,
        "name": "jobs_psid",
        "n_samples": 2490,
        "is_interventional": False,
    }

    base_url = "https://users.nber.org/~rdehejia/data/"
    data_url = base_url + "psid_controls.txt"

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        """Fetches/reads from cache the data associated with the dataset."""
        raw_data = cls._get_raw_data("data", cls.data_url)
        df = pd.read_csv(
            io.BytesIO(raw_data),
            sep=r"\s+",
            header=None,
            names=cls.columns,
        )
        for col in cls.categorical_variables:
            df[col] = df[col].astype("category")
        return df
