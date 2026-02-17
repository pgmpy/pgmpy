from .._base import PlainBIFMixin, _BaseExampleModel


class Fundraising(PlainBIFMixin, _BaseExampleModel):
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
