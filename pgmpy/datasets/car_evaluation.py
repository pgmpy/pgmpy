from pgmpy.datasets._base import BaseDataset


class CarEvaluation(BaseDataset):
    """
    References
    ----------
    - :footcite:t:`uci_car_evaluation`
    """

    _tags = {
        "name": "car_evaluation",
        "n_variables": 7,
        "n_samples": 1728,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": False,
        "is_discrete": True,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "simulated/car-evaluation"

    data_url = "data/car-evaluation.discrete.txt"
    ground_truth_url = None
    expert_knowledge_url = "ground.truth/car-evalutation.knowledge.txt"

    categorical_variables = [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "eval",
    ]
    ordinal_variables = dict()
