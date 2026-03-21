from .._base import BIFMixin, _BaseExampleModel


class Gonorrhoeae(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/gonorrhoeae",
        "n_nodes": 10,
        "n_edges": 9,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/gonorrhoeae.bif"
