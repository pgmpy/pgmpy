# This extension template provides instructions to add new datasets to pgmpy.
#
# Please follow the following steps:
# 1. Copy this file to `pgmpy/datasets` and rename the file as `your_dataset_name.py` (e.g., `my_dataset.py`).
#    Note: Do NOT start the filename with an underscore `_`, otherwise it won't be discovered.
# 2. Go through the file and address all the TODOs.
# 3. If you would like to contribute the dataset to pgmpy, please add the dataset name to ALL_DATASETS in
#   `pgmpy/tests/test_datasets/test_datasets.py` file.

import pandas

from pgmpy.base import DAG
from pgmpy.datasets._base import _BaseDataset
from pgmpy.estimators import ExpertKnowledge


class NSLM(_BaseDataset):

    _tags = {
        "name": "nslm",
        "n_variables": 11,
        "n_samples": 10391,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    data_url = "https://raw.githubusercontent.com/grf-labs/grf/refs/heads/master/experiments/acic18/synthetic_data.csv"

    ground_truth_url = None

    expert_knowledge_url = None

    missing_values_marker = None

    categorical_variables = ["C1", "C2", "C3", "XC"]

    ordinal_variables = dict()

    
