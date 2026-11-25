from pgmpy.datasets._base import BaseDataset


class NetSimSim1(BaseDataset):
    """
    References
    ----------
    - :footcite:t:`smith_2011`
    """

    _tags = {
        "name": "netsim_sim1",
        "n_variables": 5,
        "n_samples": 10000,
        "has_ground_truth": True,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "simulated/netsim"

    data_url = "data/sim1.csv"
    ground_truth_url = "ground.truth/sim1.graph.txt"
    expert_knowledge_url = None

    sep = ","

    categorical_variables = []
    ordinal_variables = dict()
