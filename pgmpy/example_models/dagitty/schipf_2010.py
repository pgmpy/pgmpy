from .._base import BaseExampleModel, DAGMixin


class Schipf2010(DAGMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`schipf_2010`
    """

    _tags = {
        "name": "dagitty/schipf_2010",
        "n_nodes": 7,
        "n_edges": 14,
        "is_parameterized": False,
    }
    data_url = "dags/Schipf_2010.txt"
