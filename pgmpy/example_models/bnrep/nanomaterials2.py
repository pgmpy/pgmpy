from .._base import BaseExampleModel, BIFMixin


class Nanomaterials2(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/nanomaterials2",
        "n_nodes": 46,
        "n_edges": 45,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/nanomaterials2.bif"
