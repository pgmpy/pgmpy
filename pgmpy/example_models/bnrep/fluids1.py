from .._base import BIFMixin, _BaseExampleModel


class Fluids1(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/fluids1",
        "n_nodes": 4,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/fluids1.bif"
