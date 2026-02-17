from .._base import PlainBIFMixin, _BaseExampleModel


class Criminal1(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/criminal1",
        "n_nodes": 12,
        "n_edges": 14,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/criminal1.bif"
