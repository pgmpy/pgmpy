from .._base import PlainBIFMixin, _BaseExampleModel


class Phdarticles(PlainBIFMixin, _BaseExampleModel):
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
