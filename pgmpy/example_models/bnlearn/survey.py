from .._base import DiscreteMixin, _BaseExampleModel


class Survey(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] M. Scutari and J.-B. Denis. Bayesian Networks: with Examples in R. Chapman & Hall, 2nd edition, 2021.
    """

    _tags = {
        "name": "bnlearn/survey",
        "n_nodes": 6,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/survey.bif.gz"
