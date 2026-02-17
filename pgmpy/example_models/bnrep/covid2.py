from .._base import BIFMixin, _BaseExampleModel


class Covid2(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/covid2",
        "n_nodes": 12,
        "n_edges": 21,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/covid2.bif"
