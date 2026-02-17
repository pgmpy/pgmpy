from .._base import BIFMixin, _BaseExampleModel


class Covidfear(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/covidfear",
        "n_nodes": 9,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/covidfear.bif"
