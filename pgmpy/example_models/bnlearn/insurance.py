from .._base import DiscreteMixin, _BaseExampleModel


class Insurance(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] J. Binder, D. Koller, S. Russell, and K. Kanazawa. Adaptive Probabilistic Networks with Hidden
    Variables. Machine Learning, 29(2-3):213-244, 1997.
    """

    _tags = {
        "name": "bnlearn/insurance",
        "n_nodes": 27,
        "n_edges": 52,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/insurance.bif.gz"
