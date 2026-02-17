from .._base import PlainBIFMixin, _BaseExampleModel


class Megacities(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/megacities",
        "n_nodes": 18,
        "n_edges": 17,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/megacities.bif"
