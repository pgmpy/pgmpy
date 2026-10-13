from .._base import ADMGMixin, BaseExampleModel


class MBias(ADMGMixin, BaseExampleModel):
    """
    References
    ----------
    - :footcite:t:`acid_decampos_1996`
    """

    _tags = {
        "name": "dagitty/m_bias",
        "n_nodes": 3,
        "n_edges": 3,
        "is_parameterized": False,
    }
    data_url = "dags/M-bias.txt"
