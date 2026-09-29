from .._base import BaseExampleModel, BIFMixin


class Witness(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/witness",
        "n_nodes": 6,
        "n_edges": 5,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/witness.bif"
