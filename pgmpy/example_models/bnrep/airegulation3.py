from .._base import PlainBIFMixin, _BaseExampleModel


class Airegulation3(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/airegulation3",
        "n_nodes": 19,
        "n_edges": 37,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/airegulation3.bif"
