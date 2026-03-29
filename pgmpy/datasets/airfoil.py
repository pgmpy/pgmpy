from pgmpy.datasets._base import _BaseDataset


class Airfoil(_BaseDataset):
    _tags = {
        "name": "airfoil",
        "n_variables": 6,
        "n_samples": 1503,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "real/airfoil-self-noise"

    data_url = "data/airfoil-self-noise.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = "ground.truth/airfoil-self-noise.knowledge.txt"

    categorical_variables = []
    ordinal_variables = dict()
