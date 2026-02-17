from .._base import PlainBIFMixin, _BaseExampleModel


class Trajectories(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/trajectories",
        "n_nodes": 5,
        "n_edges": 9,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/trajectories.bif"
