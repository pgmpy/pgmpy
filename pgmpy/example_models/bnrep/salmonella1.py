from .._base import BIFMixin, _BaseExampleModel


class Salmonella1(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/salmonella1",
        "n_nodes": 7,
        "n_edges": 5,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/salmonella1.bif"
