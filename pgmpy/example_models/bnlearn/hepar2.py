from .._base import BaseExampleModel, DiscreteMixin


class Hepar2(DiscreteMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`onisko_thesis`
    """

    _tags = {
        "name": "bnlearn/hepar2",
        "n_nodes": 70,
        "n_edges": 123,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/hepar2.bif.gz"
