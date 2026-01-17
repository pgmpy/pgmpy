from pgmpy.datasets._base import _BaseDataset


class Adult(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/adult
    """

    _tags = {
        "name": "adult",
        "n_variables": 15,
        "n_samples": 32561,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/adult/"

    data_url = base_url + "data/adult.data.mixed.maximum.50.json.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/adult.knowledge.txt"

    categorical_variables = [
        "workclass",
        "mar-stat",
        "occup",
        "relat",
        "race",
        "sex",
        "nat-count",
    ]
    ordinal_variables = {
        "educ": [
            "Preschool",
            "1st-4th",
            "5th-6th",
            "7th-8th",
            "9th",
            "10th",
            "11th",
            "12th",
            "HS-grad",
            "Some-college",
            "Assoc-voc",
            "Assoc-acdm",
            "Bachelors",
            "Masters",
            "Doctorate",
            "Prof-school",
        ],
        "Income": ["<=50K", ">50K"],
    }
