from ._base import DAGMixin, _BaseExampleModel


class Mediator(DAGMixin, _BaseExampleModel):
    """
    Simple mediator DAG (X -> I -> Y, X <- Z -> I .
    """

    _tags = {
        "name": "mediator",
        "n_nodes": 4,
        "n_edges": 5,
        "is_parameterized": False,
    }
    data_url = "dags/mediator.txt"
