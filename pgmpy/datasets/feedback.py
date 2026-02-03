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
    
    ordinal_variables = {
        "rating": [0, 1, 2, 3, 4, 5],
        "user_reputation": [1, 2, 3, 4, 5]
    }

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        """
        Loads the feedback data. We use utf-8-sig to handle 
        potential BOM markers in review CSVs.
        """
        raw_data = cls._get_raw_data("data", cls.data_url)
        return pd.read_csv(io.BytesIO(raw_data), encoding="utf-8-sig")

    @classmethod
    def load_expert_knowledge(cls) -> ExpertKnowledge:
        """
        Defines causal constraints: 
        1. Reputation -> Rating
        2. Rating -> Upvotes
        """
        if not cls.get_class_tag("has_expert_knowledge"):
            return None
        
        return ExpertKnowledge(
            required_edges=[
                ('user_reputation', 'rating'),
                ('rating', 'up_votes')
            ],
            forbidden_edges=[
                ('up_votes', 'user_reputation')
            ]
        )