from .._base import BIFMixin, _BaseExampleModel


class GDIpathway1(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/GDIpathway1",
        "n_nodes": 28,
        "n_edges": 31,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/GDIpathway1.bif"
