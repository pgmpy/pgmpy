from .._base import PlainBIFMixin, _BaseExampleModel


class Projectmanagement(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/projectmanagement",
        "n_nodes": 26,
        "n_edges": 35,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/projectmanagement.bif"
