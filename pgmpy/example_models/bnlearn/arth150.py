from .._base import ContinuousMixin, _BaseExampleModel


class arth150(ContinuousMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] R. Opgen-Rhein and K. Strimmer (2007). From Correlation to Causation Networks: a Simple Approximate Learning
    Algorithm and its Application to High-Dimensional Plant Gene Expression Data. BMC System Biology, 1(37).
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
