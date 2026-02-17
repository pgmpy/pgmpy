from .._base import PlainBIFMixin, _BaseExampleModel


class Nanomaterials1(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/nanomaterials1",
        "n_nodes": 49,
        "n_edges": 48,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/nanomaterials1.bif"
