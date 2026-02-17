from .._base import PlainBIFMixin, _BaseExampleModel


class Fingermarks2(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/fingermarks2",
        "n_nodes": 12,
        "n_edges": 20,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/fingermarks2.bif"
