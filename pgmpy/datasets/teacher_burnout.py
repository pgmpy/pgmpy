from pgmpy.datasets._base import BaseDataset


class TeacherBurnout(BaseDataset):
    """
    References
    ----------
    - :footcite:t:`byrne_2010`
    """

    _tags = {
        "name": "teacher_burnout",
        "n_variables": 32,
        "n_samples": 599,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "real/teacher-burnout"

    data_url = "data/teacher_burnout_data.csv"

    sep = ","

    categorical_variables = []

    ordinal_variables = dict()
