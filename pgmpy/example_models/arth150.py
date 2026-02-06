from ._base import ContinuousExampleMixin, _BaseExampleModel


class arth150(ContinuousExampleMixin, _BaseExampleModel):

    _tags = {
        "name": "arth150",
        "type": "continuous",
        "file_format": "json",
        "n_nodes": 107,
        "n_edges": 150,
    }
    data_url = "continuous/arth150.json"
