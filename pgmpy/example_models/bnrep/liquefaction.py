from .._base import ContinuousMixin, _BaseExampleModel


class Liquefaction(ContinuousMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/liquefaction",
        "n_nodes": 10,
        "n_edges": 15,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }

    data_url = "bnrep/liquefaction.json"
