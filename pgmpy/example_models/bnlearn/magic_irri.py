from .._base import ContinuousMixin, _BaseExampleModel


class MagicIRRI(ContinuousMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`model_2016_67`
    """

    _tags = {
        "name": "bnlearn/magic_irri",
        "n_nodes": 64,
        "n_edges": 102,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }
    data_url = "continuous/magic-irri.json"
