# This extension template provides instructions to add new datasets to pgmpy.
#
# Please follow the following steps:
# 1. Copy this file to `pgmpy/datasets` and rename the file as `_your_dataset.py`
# 2. Go through the file and address all the TODOs.
# 3. Add an import statement in the `pgmpy/datasets/__init__` file.
# 4. If you would like to contribute the dataset to pgmpy, please add the dataset name to ALL_DATASETS in
#   `pgmpy/tests/test_datasets/test_datasets.py` file.

import pandas

from pgmpy.base import DAG
from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset
from pgmpy.estimators import ExpertKnowledge


@register_dataset_class  # TODO: Rename the class for your dataset.
class YourDatasetClass(_BaseDataset):

    # TODO: Rename the name; this is the name that `load_dataset` method uses.
    name = "your_dataset_name"

    # TODO: Fill in the tags for your dataset. Expected data types are mentioned.
    tags = {
        "n_variables": int,
        "n_samples": int,
        "has_ground_truth": bool,
        "has_expert_knowledge": bool,
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

    # TODO: If the ground truth file is in dagitty format, remove the following `load_ground_truth` method.
    @classmethod
    def load_ground_truth(cls) -> DAG:
        if not cls.tags.get("has_ground_truth"):
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
        if not cls.tags.get("has_expert_knowledge"):
            return None

        _ = cls._get_raw_data("expert_knowledge", cls.expert_knowledge_url)

        # TODO: Add logic to construct a `pgmpy.estimator.ExpertKnowledge` object from data in line above.
        expert_knowledge = None
        return expert_knowledge
