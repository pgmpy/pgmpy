from .._base import BIFMixin, _BaseExampleModel


class Seismic(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/seismic",
        "n_nodes": 10,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/seismic.bif"
