from .._base import BaseExampleModel, DiscreteMixin


class Munin2(DiscreteMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`andreassen_munin`
    """

    _tags = {
        "name": "bnlearn/munin2",
        "n_nodes": 1003,
        "n_edges": 1244,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/munin2.bif.gz"
