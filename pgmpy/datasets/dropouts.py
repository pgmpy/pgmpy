# This extension template provides instructions to add new datasets to pgmpy.
#
# Please follow the following steps:
# 1. Copy this file to `pgmpy/datasets` and rename the file as `your_dataset_name.py` (e.g., `my_dataset.py`).
#    Note: Do NOT start the filename with an underscore `_`, otherwise it won't be discovered.
# 2. Go through the file and address all the TODOs.
# 3. Add an import statement in the `pgmpy/datasets/__init__.py` file (e.g. `from .my_dataset import MyDataset`).
# 4. If you would like to contribute the dataset to pgmpy, please add the dataset name to ALL_DATASETS in
#   `pgmpy/tests/test_datasets/test_datasets.py` file.

from pgmpy.datasets._base import _BaseDataset, _CovarianceMixin


class Dropout(_CovarianceMixin, _BaseDataset):

    _tags = {
        "name": "dropouts",
        "n_variables": 8,
        "n_samples": 159,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://github.com/pgmpy/example-causal-datasets/tree/main/real/dropouts"

    data_url = base_url + "/data/dropouts.cov.txt"

    ground_truth_url = None

    expert_knowledge_url = None

    missing_values_marker = None

    categorical_variables = []
    ordinal_variables = dict()

   


    
