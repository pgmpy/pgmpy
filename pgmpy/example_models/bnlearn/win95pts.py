from .._base import DiscreteMixin, _BaseExampleModel


class Win95pts(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`bnlearn_win95pts`
    """

    _tags = {
        "name": "bnlearn/win95pts",
        "n_nodes": 76,
        "n_edges": 112,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/win95pts.bif.gz"
