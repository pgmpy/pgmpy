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
        "has_index_col": False,
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

    ordinal_variables = dict()
