from .._base import BaseExampleModel, BIFMixin


class Dragline(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/dragline",
        "n_nodes": 51,
        "n_edges": 50,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/dragline.bif"
