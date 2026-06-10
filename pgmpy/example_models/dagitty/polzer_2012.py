from .._base import BaseExampleModel, DAGMixin


class Polzer2012(DAGMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`polzer_2012`
    """

    _tags = {
        "name": "dagitty/polzer_2012",
        "n_nodes": 14,
        "n_edges": 69,
        "is_parameterized": False,
    }
    data_url = "dags/Polzer_2012.txt"
