from .._base import PlainBIFMixin, _BaseExampleModel


class Covidfear(PlainBIFMixin, _BaseExampleModel):
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
