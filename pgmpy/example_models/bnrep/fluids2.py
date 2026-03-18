from .._base import BIFMixin, _BaseExampleModel


class Fluids2(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/fluids2",
        "n_nodes": 5,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/fluids2.bif"
