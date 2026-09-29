from .._base import BaseExampleModel, BIFMixin


class Arcticwaters(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/arcticwaters",
        "n_nodes": 46,
        "n_edges": 59,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/arcticwaters.bif"
