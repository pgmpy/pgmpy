import pandas as pd

from pgmpy.datasets._base import _BaseDataset


class CausalBenchDream3(_BaseDataset):
    _tags = {
        "name": "causalbench_dream3",
        "has_ground_truth": False,
        "is_simulated": True,
        "is_discrete": False,
        "is_continuous": True,
    }

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        try:
            import causalbench
        except ImportError:
            raise ImportError(
                "causalbench is required to load this dataset. Please install it using `pip install causalbench`."
            )

        return causalbench.load_dataset("Dream3")
