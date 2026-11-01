from .._base import BaseExampleModel, BIFMixin


class Vaccine(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/vaccine",
        "n_nodes": 3,
        "n_edges": 3,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/vaccine.bif"
