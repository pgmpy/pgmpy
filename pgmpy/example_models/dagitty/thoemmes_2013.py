from .._base import DAGMixin, _BaseExampleModel


class Thoemmes2013(DAGMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] Thoemmes, F. (2013). Mediation analysis with latent variables.
    """

    _tags = {
        "name": "dagitty/thoemmes_2013",
        "n_nodes": 13,
        "n_edges": 14,
        "is_parameterized": False,
    }
    data_url = "dags/Thoemmes_2013.txt"
