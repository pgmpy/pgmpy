from .._base import PlainBIFMixin, _BaseExampleModel


class Bullet(PlainBIFMixin, _BaseExampleModel):
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
