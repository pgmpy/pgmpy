from .._base import BaseExampleModel, BIFMixin


class Bullet(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/bullet",
        "n_nodes": 5,
        "n_edges": 7,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/bullet.bif"
