from .._base import BaseExampleModel, ContinuousMixin


class Ecoli70(ContinuousMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`opgenrhein_strimmer_2007`
    """

    _tags = {
        "name": "bnlearn/ecoli70",
        "n_nodes": 46,
        "n_edges": 70,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }
    data_url = "continuous/ecoli70.json"
