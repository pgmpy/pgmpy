from .._base import BaseExampleModel, DAGMixin


class Confounding(DAGMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`acid_decampos_1996`
    """

    _tags = {
        "name": "dagitty/confounding",
        "n_nodes": 5,
        "n_edges": 7,
        "is_parameterized": False,
    }
    data_url = "dags/confounding.txt"
