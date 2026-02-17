from .._base import PlainBIFMixin, _BaseExampleModel


class ConsequenceCovid(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/consequenceCovid",
        "n_nodes": 15,
        "n_edges": 34,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/consequenceCovid.bif"
