from .._base import BIFMixin, _BaseExampleModel


class Compaction(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/compaction",
        "n_nodes": 6,
        "n_edges": 5,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/compaction.bif"
