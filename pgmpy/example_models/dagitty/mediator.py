from .._base import ADMGMixin, BaseExampleModel


class Mediator(ADMGMixin, BaseExampleModel):
    """
    Simple mediator DAG (X -> I -> Y, X <- Z -> I .
    """

    _tags = {
        "name": "dagitty/mediator",
        "n_nodes": 4,
        "n_edges": 5,
        "is_parameterized": False,
    }
    data_url = "dags/mediator.txt"
