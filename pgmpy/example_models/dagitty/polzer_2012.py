from .._base import ADMGMixin, BaseExampleModel


class Polzer2012(ADMGMixin, BaseExampleModel):
    """
    References
    ----------
    - :footcite:t:`polzer_2012`
    """

    _tags = {
        "name": "dagitty/polzer_2012",
        "n_nodes": 14,
        "n_edges": 69,
        "is_parameterized": False,
    }
    data_url = "dags/Polzer_2012.txt"
