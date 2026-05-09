from .._base import DiscreteMixin, _BaseExampleModel


class Asia(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`lauritzen_spiegelhalter_1988`
    """

    _tags = {
        "name": "bnlearn/asia",
        "n_nodes": 8,
        "n_edges": 8,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/asia.bif.gz"
