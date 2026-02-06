import pandas as pd
import io
from pgmpy.datasets._base import _BaseDataset
from pgmpy.estimators import ExpertKnowledge

class Feedback(_BaseDataset):

    _tags = {
        "name": "recipe-feedback",
        "n_variables": 15,
        "n_samples": 18182,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False, 
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": True,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/main/real/recipe-feedback/"
    data_url = base_url + "data/recipe_feedback.csv"

    expert_knowledge_url = base_url + "ground.truth/feedback_knowledge.txt"

    categorical_variables = ['recipe_code', 'user_id', 'sentiment']
    
    ordinal_variables = dict()