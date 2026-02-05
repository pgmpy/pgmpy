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


# TODO: Rename the class for your dataset. If the data file is reading a covariance matrix instead of tabular data, the
# class signature should be `class YourDatasetClass(_CovarianceMixin, _BaseDataset):`.
class YourDatasetClass(_BaseDataset):

    # TODO: Fill in the tags for your dataset.
    # Note: 'name' is mandatory and must match the string used in load_dataset().
    _tags = {
        "name": "your_dataset_name",
        "n_variables": int,
        "n_samples": int,
        "has_ground_truth": bool,
        "has_expert_knowledge": bool,
        "has_missing_data": bool,
        "has_index_col": bool,
        "is_simulated": bool,
        "is_interventional": bool,
        "is_discrete": bool,
        "is_continuous": bool,
        "is_mixed": bool,
        "is_ordinal": bool,
    }

    # TODO: Add the URL to the dataset. The current parser expects the dataset to be in a tabular form with the first
    # row containing the names of the columns.
    data_url = None

    # TODO: Add the URL for the ground truth. The current parser expects the ground truth to be a dagitty model string.
    ground_truth_url = None

    # TODO: Add the URL for the expert knowledge. An example of the expected format can be found at:
    # https://github.com/pgmpy/example-causal-datasets/blob/main/real/abalone/ground.truth/abalone.knowledge.txt
    expert_knowledge_url = None

    # TODO: If the tag `has_missing_data=True`, add the marker that is used for missing values in the dataset.
    missing_values_marker = None

    # TODO: If the dataset has categorical variables, list them here.
    categorical_variables = []

    # TODO: If the dataset has ordinal variables, define the category orderings (lower to higher) for each of them here.
    ordinal_variables = dict()

    # TODO: If the ground truth file is in dagitty format, remove the following `load_ground_truth` method.
    @classmethod
    def load_ground_truth(cls) -> DAG:
        if not cls.get_class_tag("has_ground_truth"):
            return None

        _ = cls._get_raw_data("ground_truth", cls.ground_truth_url).decode(
            "utf-8-sig", errors="ignore"
        )
        # TODO: Add logic for parsing the data from the line above into a `pgmpy.base.DAG` object.
        dag = None
        return dag

    # TODO: If the data is in tabular text format, remove the following `load_dataframe` method.
    @classmethod
    def load_dataframe(cls) -> pandas.DataFrame:
        _ = cls._get_raw_data("data", cls.data_url)

        # TODO: Add logic to construct a pandas DataFrame object from data in line above.
        dataframe = None
        return dataframe

    # TODO: If the expert knowledge is in the expected format, remove the following `load_expert_knowledge` method.
    @classmethod
    def load_expert_knowledge(cls) -> ExpertKnowledge:
        if not cls.get_class_tag("has_expert_knowledge"):
            return None

        _ = cls._get_raw_data("expert_knowledge", cls.expert_knowledge_url)

        # TODO: Add logic to construct a `pgmpy.estimator.ExpertKnowledge` object from data in line above.
        expert_knowledge = None
        return expert_knowledge
