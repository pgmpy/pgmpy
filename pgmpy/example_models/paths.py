from ._base import DAGMixin, _BaseExampleModel


class Paths(DAGMixin, _BaseExampleModel):
    """
    Paths DAG from the example_models repository.
    """

    _tags = {
        "name": "paths",
        "n_nodes": 17,
        "n_edges": 19,
        "is_parameterized": False,
    }
    data_url = "dags/paths.txt"
