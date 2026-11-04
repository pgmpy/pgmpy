from pgmpy.datasets._base import BaseDataset


class CausalBenchK562(BaseDataset):
    """
    References
    ----------
    - :footcite:t:`chevalley_2025`
    """

    _tags = {
        "name": "causalbench_k562",
        "n_variables": 622,
        "n_samples": 162751,
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

    repo_id = "huggingbahl21/causalbench"
    base_url = "data/prepared"

    data_url = "causalbench_k562.tsv.gz"
    ground_truth_url = None
    expert_knowledge_url = None

    compression = "gzip"
    intervention_columns = ["intervention"]
    categorical_variables = ["intervention"]
    ordinal_variables = dict()
