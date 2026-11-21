from pgmpy.datasets._base import BaseDataset


class GeneralSocialSurvey(BaseDataset):
    """
    Occupational attainment data from the General Social Survey, used by :footcite:t:`shimizu_2011` to evaluate
    DirectLiNGAM.

    The columns are ``v1``: father's occupation level, ``v2``: son's income, ``v3``: father's education, ``v4``: son's
    occupation level, ``v5``: son's education and ``v6``: number of siblings.

    References
    ----------
    - :footcite:t:`shimizu_2011`
    """

    _tags = {
        "name": "general_social_survey",
        "n_variables": 6,
        "n_samples": 1380,
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

    base_url = "real/general-social-survey"

    data_url = "data/general_social_survey.csv"

    sep = ","

    categorical_variables = []

    ordinal_variables = dict()
