from ._base import DiscreteExampleMixin, _BaseExampleModel


class alarm(DiscreteExampleMixin, _BaseExampleModel):

    _tags = {
        "name": "alarm",
        "type": "discrete",
        "file_format": "bif",
        "n_nodes": 37,
        "n_edges": 46,
    }
    data_url = "discrete/alarm.bif.gz"
