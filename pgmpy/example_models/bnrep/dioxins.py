from .._base import BIFMixin, _BaseExampleModel


class Dioxins(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/dioxins",
        "n_nodes": 9,
        "n_edges": 15,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/dioxins.bif"
