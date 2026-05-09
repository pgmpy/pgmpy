from .._base import DAGMixin, _BaseExampleModel


class Confounding(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`acid_1996_54`
    """

    _tags = {
        "name": "dagitty/confounding",
        "n_nodes": 5,
        "n_edges": 7,
        "is_parameterized": False,
    }
    data_url = "dags/confounding.txt"
