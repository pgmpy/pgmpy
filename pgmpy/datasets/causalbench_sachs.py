import pandas as pd

from pgmpy.datasets._base import _BaseDataset


class CausalBenchSachs(_BaseDataset):
    _tags = {
        "name": "causalbench_sachs",
        "has_ground_truth": True,
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

        # Load the dataset using causalbench API
        return causalbench.load_dataset("Sachs")

    @classmethod
    def load_ground_truth(cls):
        # Optional ground truth retrieval logic
        # return causalbench.load_ground_truth("Sachs")
        return None
