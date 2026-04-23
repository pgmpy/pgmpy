from .._base import BIFMixin, _BaseExampleModel


class Adhd(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/adhd",
        "n_nodes": 19,
        "n_edges": 18,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/adhd.bif"
