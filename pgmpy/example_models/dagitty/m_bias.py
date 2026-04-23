from .._base import DAGMixin, _BaseExampleModel


class MBias(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] Acid, S., & De Campos, L. M. (1996). An algorithm for finding minimum d-separating sets in belief networks.
    Proceedings of the Twelfth International Conference on Uncertainty in Artificial Intelligence, 3–10. Presented at
    the Portland, OR. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.
    """

    _tags = {
        "name": "dagitty/m_bias",
        "n_nodes": 5,
        "n_edges": 5,
        "is_parameterized": False,
    }
    data_url = "dags/M-bias.txt"
