from pgmpy.datasets._base import _BaseDataset
import pandas as pd

class CausalBenchDream3(_BaseDataset):
    _tags = {
        "name": "causalbench_dream3",
        "has_ground_truth": True,
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
        
        # Load the dataset using causalbench API
        return causalbench.load_dataset("Dream3")

    @classmethod
    def load_ground_truth(cls):
        try:
            import causalbench
        except ImportError:
            raise ImportError(
                "causalbench is required to load this dataset. Please install it using `pip install causalbench`."
            )
            
        # Optional ground truth retrieval logic
        return None
