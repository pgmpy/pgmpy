from .._base import BaseExampleModel, DiscreteMixin


class Cancer(DiscreteMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`korb_nicholson_2010`
    """

    _tags = {
        "name": "bnlearn/cancer",
        "n_nodes": 5,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/cancer.bif.gz"
