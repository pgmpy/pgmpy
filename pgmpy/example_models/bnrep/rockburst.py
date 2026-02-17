from .._base import PlainBIFMixin, _BaseExampleModel


class Rockburst(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/rockburst",
        "n_nodes": 6,
        "n_edges": 9,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/rockburst.bif"
