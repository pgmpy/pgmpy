from pgmpy.datasets._base import _BaseDataset


class CarEvaluation(_BaseDataset):
    """
    References
    ----------
    .. [1] Bohanec, M. (1988). Car Evaluation. UCI Machine Learning Repository.
           https://doi.org/10.24432/C5JP48
    .. [2] https://github.com/pgmpy/example-causal-datasets
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
        "is_ordinal": True,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/main/simulated/car-evaluation/"

    data_url = base_url + "data/car-evaluation.discrete.txt"
    ground_truth_url = None
    expert_knowledge_url = (
        base_url + "ground.truth/car-evalutation.knowledge.txt"
    )

    categorical_variables = []
    ordinal_variables = {
        "buying": ["low", "med", "high", "vhigh"],
        "maint": ["low", "med", "high", "vhigh"],
        "doors": ["2", "3", "4", "5more"],
        "persons": ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety": ["low", "med", "high"],
        "eval": ["unacc", "acc", "good", "vgood"],
    }
