from .._base import PlainBIFMixin, _BaseExampleModel


class Covidtech(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/covidtech",
        "n_nodes": 18,
        "n_edges": 18,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/covidtech.bif"
