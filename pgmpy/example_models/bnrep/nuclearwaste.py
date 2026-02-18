from .._base import BIFMixin, _BaseExampleModel


class Nuclearwaste(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/nuclearwaste",
        "n_nodes": 10,
        "n_edges": 11,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/nuclearwaste.bif"
