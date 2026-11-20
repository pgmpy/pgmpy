from pgmpy.datasets._base import BaseDataset


class TeacherBurnout(BaseDataset):
    """
    Teacher Burnout dataset from Byrne (2010) [1]_, sourced via [2]_.

    References
    ----------
    ..[1] Byrne, B. M. (2010). Structural Equation Modeling with AMOS:
           Basic Concepts, Applications, and Programming (2nd ed.). Routledge.
           https://doi.org/10.4324/9780203805534
    ..[2] https://github.com/kurowasan/cd_datasets/blob/main/data/teacher_burnout_data.csv
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

    ground_truth_url = None

    expert_knowledge_url = None

    missing_values_marker = None

    sep = ","

    intervention_columns = None

    categorical_variables = []

    ordinal_variables = dict()
