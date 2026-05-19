from .._base import DAGMixin, _BaseExampleModel


class MBias(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`acid_decampos_1996`
    """

    _tags = {
        "name": "dagitty/m_bias",
        "n_nodes": 5,
        "n_edges": 5,
        "is_parameterized": False,
    }
    data_url = "dags/M-bias.txt"
