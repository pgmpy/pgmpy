from ._base import ContinuousMixin, _BaseExampleModel


class arth150(ContinuousMixin, _BaseExampleModel):

    _tags = {
        "name": "arth150",
        "type": "continuous",
        "file_format": "json",
        "n_nodes": 107,
        "n_edges": 150,
    }
    data_url = "continuous/arth150.json"
