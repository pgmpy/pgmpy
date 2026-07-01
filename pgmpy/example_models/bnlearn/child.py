from .._base import BaseExampleModel, DiscreteMixin


class Child(DiscreteMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`spiegelhalter_cowell_1992`
    """

    _tags = {
        "name": "bnlearn/child",
        "n_nodes": 20,
        "n_edges": 25,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/child.bif.gz"
