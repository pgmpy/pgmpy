import pandas as pd

from pgmpy.datasets._base import _BaseDataset


class CausalBenchSachs(_BaseDataset):
    _tags = {
        "name": "causalbench_sachs",
        "has_ground_truth": False,
        "is_simulated": False,
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

        return causalbench.load_dataset("Sachs")
