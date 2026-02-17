from .._base import PlainBIFMixin, _BaseExampleModel


class Soillead(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/soillead",
        "n_nodes": 9,
        "n_edges": 22,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/soillead.bif"
