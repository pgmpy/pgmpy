from .._base import BaseExampleModel, BIFMixin


class Pneumonia(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/pneumonia",
        "n_nodes": 62,
        "n_edges": 171,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/pneumonia.bif"
