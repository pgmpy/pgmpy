from pgmpy.datasets._base import _BaseDataset


class AngristKruegerCPS(_BaseDataset):
    """
    References
    ----------
    .. [1] Angrist, J. D., & Krueger, A. B. (1995). Split-Sample Instrumental Variables Estimates
           of the Return to Schooling. Journal of Business & Economic Statistics, 13(2), 225-235.
    """

    _tags = {
        "name": "angrist_krueger_cps",
        "n_variables": 58,
        "n_samples": 13993,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/Rasesh2005/example_datasets/feature/angrist-kreuger-dataset/"
        "real/angrist-krueger-cps/"
    )

    data_url = base_url + "data/angrist-krueger-cps.mixed.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = [
        "lot1b50",
        "lot2b50",
        "lot3b50",
        "lot1b51",
        "lot2b51",
        "lot3b51",
        "lot1b52",
        "lot2b52",
        "lot3b52",
        "lot1b53",
        "lot2b53",
        "lot3b53",
        "midatl",
        "eastnth",
        "westnth",
        "black",
        "other",
        "city",
        "balsmsa",
        "spsepres",
        "yr81",
        "yr82",
        "yr83",
        "yr84",
        "yr85",
        "yob45",
        "yob46",
        "yob47",
        "yob48",
        "yob49",
        "yob50",
        "yob51",
        "yob52",
        "yob53",
        "educ",
        "veteran",
        "ceiling",
        "coarse1",
        "coarse2",
        "coarse3",
        "lott1",
        "lott2",
        "lott3",
        "lott4",
        "lott5",
        "lott6",
        "lott7",
        "lott8",
        "lott9",
        "lott10",
        "lott11",
        "lott12",
        "lott13",
        "recode",
    ]
    ordinal_variables = dict()
