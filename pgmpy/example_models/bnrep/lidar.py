from .._base import PlainBIFMixin, _BaseExampleModel


class Lidar(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/lidar",
        "n_nodes": 9,
        "n_edges": 8,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/lidar.bif"
