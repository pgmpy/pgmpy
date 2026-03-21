from .._base import DiscreteMixin, _BaseExampleModel


class Hepar2(DiscreteMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] A. Onisko. Probabilistic Causal Models in Medicine: Application to Diagnosis of Liver Disorders. Ph.D.
    Dissertation, Institute of Biocybernetics and Biomedical Engineering, Polish Academy of Science, Warsaw, March
    2003.
    """

    _tags = {
        "name": "bnlearn/hepar2",
        "n_nodes": 70,
        "n_edges": 123,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/hepar2.bif.gz"
