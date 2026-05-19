from .._base import DiscreteMixin, _BaseExampleModel


class Hailfinder(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`abramson_1996`
    """

    _tags = {
        "name": "bnlearn/hailfinder",
        "n_nodes": 56,
        "n_edges": 66,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/hailfinder.bif.gz"
