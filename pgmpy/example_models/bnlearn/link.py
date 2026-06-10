from .._base import BaseExampleModel, DiscreteMixin


class Link(DiscreteMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`jensen_kong`
    """

    _tags = {
        "name": "bnlearn/link",
        "n_nodes": 724,
        "n_edges": 1125,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/link.bif.gz"
