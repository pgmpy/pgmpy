from .._base import PlainBIFMixin, _BaseExampleModel


class Criminal3(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/criminal3",
        "n_nodes": 8,
        "n_edges": 9,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/criminal3.bif"
