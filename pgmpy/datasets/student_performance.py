from pgmpy.datasets._base import _BaseDataset


class StudentPerformance(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/student+performance
    """

    _tags = {
        "name": "student_performance",
        "n_variables": 33,
        "n_samples": 395,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/student-performance/"

    data_url = base_url + "data/student-performance.data.mixed.maximum.3.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/student-performance.knowledge.txt"

    categorical_variables = [
        "school",
        "sex",
        "address",
        "famsize",
        "Pstatus",
        "Mjob",
        "Fjob",
        "reason",
        "guardian",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
        "G1",
        "G2",
    ]

    ordinal_variables = {
        "Medu": [0, 1, 2, 3, 4],
        "Fedu": [0, 1, 2, 3, 4],
        "traveltime": [1, 2, 3, 4],
        "studytime": [1, 2, 3, 4],
        "failures": [1, 2, 3, 4],
        "famrel": [1, 2, 3, 4, 5],
        "freetime": [1, 2, 3, 4, 5],
        "goout": [1, 2, 3, 4, 5],
        "Dalc": [1, 2, 3, 4, 5],
        "Walc": [1, 2, 3, 4, 5],
        "health": [1, 2, 3, 4, 5],
    }
