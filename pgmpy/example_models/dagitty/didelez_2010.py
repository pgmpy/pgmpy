from .._base import DAGMixin, _BaseExampleModel


class Didelez2010(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`didelez_2010`
    """

    _tags = {
        "name": "dagitty/didelez_2010",
        "n_nodes": 7,
        "n_edges": 11,
        "is_parameterized": False,
    }
    data_url = "dags/Didelez_2010.txt"
