from .._base import BIFMixin, _BaseExampleModel


class Moodstate(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/moodstate",
        "n_nodes": 7,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/moodstate.bif"
