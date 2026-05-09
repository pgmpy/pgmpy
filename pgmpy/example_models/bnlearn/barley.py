from .._base import DiscreteMixin, _BaseExampleModel


class Barley(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`preliminary_noyear_70`
    """

    _tags = {
        "name": "bnlearn/barley",
        "n_nodes": 48,
        "n_edges": 84,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/barley.bif.gz"
