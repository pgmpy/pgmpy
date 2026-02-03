import pandas as pd
import io
from pgmpy.datasets._base import _BaseDataset

class CarEvaluation(_BaseDataset):

    _tags = {
        "name":  "car-evaluation", 
        "n_variables": 7,
        "n_samples": 1728,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": True,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": True,
    }

    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    
    ground_truth_url = None
    expert_knowledge_url = None
    missing_values_marker = None


    categorical_variables = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

    ordinal_variables = {
        "buying": ["low", "med", "high", "vhigh"],
        "maint": ["low", "med", "high", "vhigh"],
        "doors": ["2", "3", "4", "5more"],
        "persons": ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety": ["low", "med", "high"],
        "class": ["unacc", "acc", "good", "vgood"]
    }

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        """
        Custom loader because the raw UCI file lacks a header row.
        """
        raw_data = cls._get_raw_data("data", cls.data_url)
        data_str = raw_data.decode("utf-8")
        
        column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        
        df = pd.read_csv(io.StringIO(data_str), names=column_names, header=None)
        return df

    