from .._base import PlainBIFMixin, _BaseExampleModel


class Grounding(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/grounding",
        "n_nodes": 36,
        "n_edges": 45,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/grounding.bif"
