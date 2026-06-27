from .._base import BaseExampleModel, DAGMixin


class Shrier2008(DAGMixin, BaseExampleModel):
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
