from .._base import BIFMixin, _BaseExampleModel


class Soillead(BIFMixin, _BaseExampleModel):
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
