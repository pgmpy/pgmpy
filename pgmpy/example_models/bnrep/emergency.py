from .._base import PlainBIFMixin, _BaseExampleModel


class Emergency(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/emergency",
        "n_nodes": 17,
        "n_edges": 16,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/emergency.bif"
