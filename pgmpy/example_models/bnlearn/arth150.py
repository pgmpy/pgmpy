from .._base import BaseExampleModel, ContinuousMixin


class Arth150(ContinuousMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`opgenrhein_strimmer_2007`
    """

    _tags = {
        "name": "bnlearn/arth150",
        "n_nodes": 107,
        "n_edges": 150,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }
    data_url = "continuous/arth150.json"
