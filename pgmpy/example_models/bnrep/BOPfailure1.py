from .._base import PlainBIFMixin, _BaseExampleModel


class BOPfailure1(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/BOPfailure1",
        "n_nodes": 30,
        "n_edges": 29,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/BOPfailure1.bif"
