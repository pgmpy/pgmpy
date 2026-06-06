from .._base import BaseExampleModel, DiscreteMixin


class Pigs(DiscreteMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`bnlearn_pigs`
    """

    _tags = {
        "name": "bnlearn/pigs",
        "n_nodes": 441,
        "n_edges": 592,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/pigs.bif.gz"
