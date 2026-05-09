from .._base import DiscreteMixin, _BaseExampleModel


class Insurance(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`binder_1997`
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
