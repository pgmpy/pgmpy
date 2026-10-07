from .._base import ADMGMixin, BaseExampleModel


class Paths(ADMGMixin, BaseExampleModel):
    """
    Paths DAG from the example_models repository.
    """

    _tags = {
        "name": "dagitty/paths",
        "n_nodes": 17,
        "n_edges": 19,
        "is_parameterized": False,
    }
    data_url = "dags/paths.txt"
