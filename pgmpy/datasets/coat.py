from pgmpy.datasets._base import BaseDataset


class Coat(BaseDataset):
    """Coat shopping dataset: self-selected product ratings for causal recommendation.

    290 users rated a subset of 300 coat items on a 1-5 scale; each user rated
    exactly 24 self-selected items. Unobserved entries therefore reflect
    selection bias rather than random missingness, which makes the dataset a
    standard benchmark for debiasing learning and evaluation of recommender
    systems under missing-not-at-random (MNAR) data, e.g. with inverse
    propensity scoring.

    Source: http://www.cs.cornell.edu/~schnabts/mnar/
    Original archive: ``coat.zip`` (537,499 bytes).
    Packaged file: the self-selected training ratings (``train.ascii``)
    converted to tab-separated form with a header row; unrated entries are
    stored as ``NA``.
    License: CC BY-NC 4.0.

    References
    ----------
    - :footcite:t:`schnabel_2016`
    - :footcite:t:`schnabel_2016_coat_dataset`
    """

    _tags = {
        "name": "coat",
        "n_variables": 300,
        "n_samples": 290,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": True,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": True,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "real/coat"

    data_url = "data/coat.train.discrete.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    missing_values_marker = "NA"

    categorical_variables = []
    ordinal_variables = dict()
