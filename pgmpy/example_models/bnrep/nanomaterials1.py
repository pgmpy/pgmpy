from .._base import BaseExampleModel, BIFMixin


class Nanomaterials1(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/nanomaterials1",
        "n_nodes": 49,
        "n_edges": 48,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/nanomaterials1.bif"
