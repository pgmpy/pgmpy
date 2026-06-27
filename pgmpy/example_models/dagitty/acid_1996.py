from .._base import BaseExampleModel, DAGMixin


class Acid1996(DAGMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`acid_decampos_1996`
    """

    _tags = {
        "name": "dagitty/acid_1996",
        "n_nodes": 18,
        "n_edges": 22,
        "is_parameterized": False,
    }
    data_url = "dags/Acid_1996.txt"
