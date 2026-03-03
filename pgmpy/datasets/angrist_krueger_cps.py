from pgmpy.datasets._base import _BaseDataset


class AngristKruegerCPS(_BaseDataset):
    """
    The Angrist-Krueger CPS dataset is a subset of the Current Population
    Survey (CPS) used for investigating the return to schooling using
    split-sample instrumental variables. It contains 13,993 observations
    and 58 variables.

    This is a 'mixed' dataset containing both continuous and discrete
    variables. Continuous variables include annual earnings (annwage),
    weekly wages (wkwage), and their natural logarithms (lnyrwage,
    lnwkwage).

    Categorical variables include demographic indicators (e.g., black,
    other, city), year indicators, and various interaction terms between
    cohorts and regions (e.g., lot1b50, lott1-lott13). These are
    explicitly typed as categorical to ensure statistical models treat
    them as factors rather than continuous values.

    References
    ----------
    .. [1] Angrist, J. D., & Krueger, A. B. (1995). Split-Sample Instrumental
           Variables Estimates of the Return to Schooling. Journal of Business
           & Economic Statistics, 13(2), 225-235.
           Source: https://economics.mit.edu/people/faculty/josh-angrist/
           angrist-data-archive
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
        "https://raw.githubusercontent.com/Rasesh2005/example_datasets/"
        "feature/angrist-kreuger-dataset/real/angrist-krueger-cps/"
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
