from pgmpy.datasets._base import BaseDataset


class CausalBenchRPE1(BaseDataset):
    """
    References
    ----------
    - :footcite:t:`chevalley_2025`
    """

    _tags = {
        "name": "causalbench_rpe1",
        "n_variables": 383,
        "n_samples": 162733,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "real/causalbench"

    data_url = "data/causalbench_rpe1.tsv.gz"
    ground_truth_url = None
    expert_knowledge_url = None

    compression = "gzip"
    intervention_columns = ["intervention"]
    categorical_variables = ["intervention"]
    ordinal_variables = dict()
