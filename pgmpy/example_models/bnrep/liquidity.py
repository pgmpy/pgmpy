from .._base import PlainBIFMixin, _BaseExampleModel


class Liquidity(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/liquidity",
        "n_nodes": 10,
        "n_edges": 13,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/liquidity.bif"
