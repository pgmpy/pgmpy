from .._base import DiscreteMixin, _BaseExampleModel


class Mildew(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`jensen_jensen_midas`
    """

    _tags = {
        "name": "bnlearn/mildew",
        "n_nodes": 35,
        "n_edges": 46,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/mildew.bif.gz"
