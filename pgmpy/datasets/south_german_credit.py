from pgmpy.datasets._base import _BaseDataset


class SouthGermanCredit(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29
    """

    _tags = {
        "name": "south_german_credit",
        "n_variables": 21,
        "n_samples": 1000,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/south-german-credit/"

    data_url = base_url + "data/south-german-credit.data.mixed.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = [
        "status",
        "credit_history",
        "purpose",
        "savings",
        "personal_status_sex",
        "other_debtors",
        "other_installment_plans",
        "housing",
        "people_liable",
        "telephone",
        "foreign_worker",
        "credit_risk",
    ]
    ordinal_variables = {
        "employment_duration": [1, 2, 3, 4, 5],
        "installment_rate": [1, 2, 3, 4],
        "present_residence": [1, 2, 3, 4],
        "property": [1, 2, 3, 4],
        "number_credits": [1, 2, 3, 4],
        "job": [1, 2, 3, 4],
    }
