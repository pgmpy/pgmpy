from .._base import BIFMixin, _BaseExampleModel


class Greencredit(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/greencredit",
        "n_nodes": 10,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/greencredit.bif"
