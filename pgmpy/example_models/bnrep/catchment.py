from .._base import BIFMixin, _BaseExampleModel


class Catchment(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/catchment",
        "n_nodes": 19,
        "n_edges": 26,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/catchment.bif"
