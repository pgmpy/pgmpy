from .._base import BIFMixin, _BaseExampleModel


class Soilliquefaction3(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/soilliquefaction3",
        "n_nodes": 7,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/soilliquefaction3.bif"
