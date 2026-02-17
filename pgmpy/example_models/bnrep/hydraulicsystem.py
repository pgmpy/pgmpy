from .._base import BIFMixin, _BaseExampleModel


class Hydraulicsystem(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/hydraulicsystem",
        "n_nodes": 4,
        "n_edges": 3,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/hydraulicsystem.bif"
