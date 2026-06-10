from .._base import BaseExampleModel, DiscreteMixin


class Munin(DiscreteMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`andreassen_munin`
    """

    _tags = {
        "name": "bnlearn/munin",
        "n_nodes": 1041,
        "n_edges": 1397,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/munin.bif.gz"
