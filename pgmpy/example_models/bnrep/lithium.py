from .._base import BIFMixin, _BaseExampleModel


class Lithium(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/lithium",
        "n_nodes": 45,
        "n_edges": 44,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/lithium.bif"
