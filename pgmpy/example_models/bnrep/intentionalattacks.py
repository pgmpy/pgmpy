from .._base import BaseExampleModel, BIFMixin


class Intentionalattacks(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/intentionalattacks",
        "n_nodes": 8,
        "n_edges": 7,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/intentionalattacks.bif"
