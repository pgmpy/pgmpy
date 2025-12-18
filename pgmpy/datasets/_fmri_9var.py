import io
from typing import List

import pandas as pd

from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class FMRI9Var(_BaseDataset):
    name = "fmri_9var"

    tags = {
        "n_variables": 9,
        "n_samples": 1440,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/fmri.9var/"
    )

    _subject_files: List[str] = [
        "sub001.cent-table.9.continuous.txt",
        "sub004.cent-table.9.continuous.txt",
        "sub005.cent-table.9.continuous.txt",
        "sub009.cent-table.9.continuous.txt",
        "sub010.cent-table.9.continuous.txt",
        "sub013.cent-table.9.continuous.txt",
        "sub014.cent-table.9.continuous.txt",
        "sub016.cent-table.9.continuous.txt",
        "sub017.cent-table.9.continuous.txt",
    ]

    data_url = None
    ground_truth_url = None
    expert_knowledge_url = None

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        dfs = []
        for fname in cls._subject_files:
            url = cls.base_url + "data/9var/" + fname
            raw = cls._get_raw_data(f"data_{fname}", url)
            df = pd.read_csv(io.BytesIO(raw), sep="\t")
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)
