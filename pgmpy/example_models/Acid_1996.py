from ._base import DAGMixin, _BaseExampleModel


class Acid_1996(DAGMixin, _BaseExampleModel):

    _tags = {
        "name": "Acid_1996",
        "n_nodes": 18,
        "n_edges": 24,
        "is_parameterized": False,
    }
    data_url = "dags/Acid_1996.txt"
