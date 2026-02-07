from ._base import DAGMixin, _BaseExampleModel


class Kampen2014(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] van Kampen, et al. (2014). Causal DAG for epidemiological analysis.
    """

    _tags = {
        "name": "kampen_2014",
        "n_nodes": 12,
        "n_edges": 24,
        "is_parameterized": False,
    }
    data_url = "dags/Kampen_2014.txt"
