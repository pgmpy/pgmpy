from ._base import DAGMixin, _BaseExampleModel


class Schipf2010(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] Schipf, S., et al. (2010). Causal DAG for type 2 diabetes and related factors.
    """

    _tags = {
        "name": "schipf_2010",
        "n_nodes": 7,
        "n_edges": 14,
        "is_parameterized": False,
    }
    data_url = "dags/Schipf_2010.txt"
