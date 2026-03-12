from .._base import BIFMixin, _BaseExampleModel


class Blockchain(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/blockchain",
        "n_nodes": 12,
        "n_edges": 13,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/blockchain.bif"
