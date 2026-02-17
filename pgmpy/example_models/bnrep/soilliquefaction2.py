from .._base import BIFMixin, _BaseExampleModel


class Soilliquefaction2(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/soilliquefaction2",
        "n_nodes": 7,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/soilliquefaction2.bif"
