from .._base import PlainBIFMixin, _BaseExampleModel


class Coral2(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/coral2",
        "n_nodes": 8,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/coral2.bif"
