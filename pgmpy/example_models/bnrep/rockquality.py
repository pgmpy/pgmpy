from .._base import BIFMixin, _BaseExampleModel


class Rockquality(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/rockquality",
        "n_nodes": 9,
        "n_edges": 8,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/rockquality.bif"
