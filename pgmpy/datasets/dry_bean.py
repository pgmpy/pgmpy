from pgmpy.datasets._base import BaseDataset


class DryBean(BaseDataset):
    """
    References
    ----------
    - :cite:p:`uci_dry_bean`
    """

    _tags = {
        "name": "dry_bean",
        "n_variables": 17,
        "n_samples": 13611,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "real/dry-bean"

    data_url = "data/drybean.data.mixed.maximum.7.txt"
    ground_truth_url = None
    expert_knowledge_url = "ground.truth/dry-bean.knowledge.txt"

    categorical_variables = ["Class"]
    ordinal_variables = dict()
