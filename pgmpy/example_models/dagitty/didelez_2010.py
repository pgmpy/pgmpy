from .._base import ADMGMixin, BaseExampleModel


class Didelez2010(ADMGMixin, BaseExampleModel):
    """
    References
    ----------
    - :footcite:t:`didelez_2010`
    """

    _tags = {
        "name": "dagitty/didelez_2010",
        "n_nodes": 7,
        "n_edges": 11,
        "is_parameterized": False,
    }
    data_url = "dags/Didelez_2010.txt"
