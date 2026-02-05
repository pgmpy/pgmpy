from ._base import DAGExampleMixin, _BaseExampleModel


class Acid1996(DAGExampleMixin, _BaseExampleModel):

    _tags = {
        "name": "Acid_1996",
        "type": "dags",
        "file_format": "txt",
        "n_nodes": 18,
        "n_edges": 24,
    }
    data_url = "dags/Acid_1996.txt"
