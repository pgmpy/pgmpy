from .._base import DiscreteMixin, _BaseExampleModel


class Munin3(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`andreassen_munin`
    """

    _tags = {
        "name": "bnlearn/munin3",
        "n_nodes": 1041,
        "n_edges": 1306,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/munin3.bif.gz"
