from .._base import PlainBIFMixin, _BaseExampleModel


class BOPfailure2(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/BOPfailure2",
        "n_nodes": 46,
        "n_edges": 45,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/BOPfailure2.bif"
