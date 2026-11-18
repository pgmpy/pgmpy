from .._base import BaseExampleModel, BIFMixin


class Fundraising(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/fundraising",
        "n_nodes": 8,
        "n_edges": 8,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/fundraising.bif"
