from .._base import PlainBIFMixin, _BaseExampleModel


class Perioperative(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/perioperative",
        "n_nodes": 5,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/perioperative.bif"
