from .._base import PlainBIFMixin, _BaseExampleModel


class Pneumonia(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/pneumonia",
        "n_nodes": 62,
        "n_edges": 171,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/pneumonia.bif"
