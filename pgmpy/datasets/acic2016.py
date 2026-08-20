from pgmpy.datasets._base import BaseDataset


class ACIC2016(BaseDataset):
    """
    Atlantic Causal Inference Conference (ACIC) 2016 benchmark dataset.
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

    base_url = "simulated/acic-2016"

    data_url = "data/acic.2016.sim.1.mixed.txt"

    ground_truth_url = None

    expert_knowledge_url = None

    categorical_variables = ["x2", "x21", "x24"]

    ordinal_variables = {}
