from .._base import DAGMixin, _BaseExampleModel


class Shrier2008(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`shrier_2008`
    """

    _tags = {
        "name": "dagitty/shrier_2008",
        "n_nodes": 13,
        "n_edges": 19,
        "is_parameterized": False,
    }
    data_url = "dags/Shrier_2008.txt"
