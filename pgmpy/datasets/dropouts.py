from pgmpy.datasets._base import _BaseDataset, _CovarianceMixin


class Dropouts(_CovarianceMixin, _BaseDataset):

    _tags = {
        "name": "dropouts",
        "n_variables": 8,
        "n_samples": 159,
        "has_ground_truth": False,
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

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example_datasets/"
        "refs/heads/main/real/dropouts/"
    )

    data_url = base_url + "data/dropouts.cov.txt"

    ground_truth_url = None

    expert_knowledge_url = None
