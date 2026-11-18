from .._base import BaseExampleModel, BIFMixin


class Curacao4(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/curacao4",
        "n_nodes": 13,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/curacao4.bif"
