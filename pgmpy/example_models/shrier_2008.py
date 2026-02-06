from ._base import DAGMixin, _BaseExampleModel


class Shrier2008(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] Shrier, I. (2008). Understanding the causal effect of warm-up on injury risk.
    British Journal of Sports Medicine, 42(10), 852–853.
    """

    _tags = {
        "name": "shrier_2008",
        "n_nodes": 13,
        "n_edges": 19,
        "is_parameterized": False,
    }
    data_url = "dags/Shrier_2008.txt"
