from .._base import DAGMixin, _BaseExampleModel


class Kampen2014(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`vankampen_2014`
    """

    _tags = {
        "name": "dagitty/kampen_2014",
        "n_nodes": 12,
        "n_edges": 24,
        "is_parameterized": False,
    }
    data_url = "dags/Kampen_2014.txt"
