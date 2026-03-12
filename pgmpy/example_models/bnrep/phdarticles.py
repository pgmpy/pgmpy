from .._base import BIFMixin, _BaseExampleModel


class Phdarticles(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/phdarticles",
        "n_nodes": 6,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/phdarticles.bif"
