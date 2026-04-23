from .._base import BIFMixin, _BaseExampleModel


class Soil(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/soil",
        "n_nodes": 6,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/soil.bif"
