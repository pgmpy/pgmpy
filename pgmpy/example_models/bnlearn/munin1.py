from .._base import DiscreteMixin, _BaseExampleModel


class Munin1(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`andreassen_munin`
    """

    _tags = {
        "name": "bnlearn/munin1",
        "n_nodes": 186,
        "n_edges": 273,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/munin1.bif.gz"
