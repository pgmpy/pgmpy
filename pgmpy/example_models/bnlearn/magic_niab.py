from .._base import ContinuousMixin, _BaseExampleModel


class MagicNIAB(ContinuousMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`opgenrhein_strimmer_2007`
    """

    _tags = {
        "name": "bnlearn/magic_niab",
        "n_nodes": 44,
        "n_edges": 66,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }
    data_url = "continuous/magic-niab.json"
