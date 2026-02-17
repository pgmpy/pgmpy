from .._base import BIFMixin, _BaseExampleModel


class Projectmanagement(BIFMixin, _BaseExampleModel):
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
