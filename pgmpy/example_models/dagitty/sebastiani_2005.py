from .._base import BaseExampleModel, DAGMixin


class Sebastiani2005(DAGMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`sebastiani_2005`
    """

    _tags = {
        "name": "dagitty/sebastiani_2005",
        "n_nodes": 36,
        "n_edges": 60,
        "is_parameterized": False,
    }
    data_url = "dags/Sebastiani_2005.txt"
