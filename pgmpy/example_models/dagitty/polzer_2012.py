from .._base import DAGMixin, _BaseExampleModel


class Polzer2012(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] Polzer, I., et al. (2012). Causal DAG for periodontitis / tooth loss and mortality.
    """

    _tags = {
        "name": "dagitty/polzer_2012",
        "n_nodes": 14,
        "n_edges": 69,
        "is_parameterized": False,
    }
    data_url = "dags/Polzer_2012.txt"
