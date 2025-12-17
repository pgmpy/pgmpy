import io
import pandas as pd

from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class Hitters(_BaseDataset):
    name = "hitters"

    tags = {
        "n_variables": 20,
        "n_samples": 322,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/hitters/"
    )

    data_url = base_url + "data/hitters.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        raw = cls._get_raw_data("data", cls.data_url).decode("utf-8-sig", errors="ignore")

        df = pd.read_csv(
            io.StringIO(raw),
            sep=r"\s+",
            engine="python",
            na_values=["*"],
        )

        if "Salary" in df.columns:
            df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

        return df
