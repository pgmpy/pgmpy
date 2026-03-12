from .._base import BIFMixin, _BaseExampleModel


class Metal(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/metal",
        "n_nodes": 8,
        "n_edges": 7,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/metal.bif"
