from .._base import BaseExampleModel, DiscreteMixin


class Sachs(DiscreteMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`sachs_2005`
    """

    _tags = {
        "name": "bnlearn/sachs",
        "n_nodes": 11,
        "n_edges": 17,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/sachs.bif.gz"
