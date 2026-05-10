from .._base import DiscreteMixin, _BaseExampleModel


class Munin4(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`andreassen_munin`
    """

    _tags = {
        "name": "bnlearn/munin4",
        "n_nodes": 1038,
        "n_edges": 1388,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/munin4.bif.gz"
