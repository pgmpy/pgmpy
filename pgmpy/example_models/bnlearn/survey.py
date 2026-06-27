from .._base import BaseExampleModel, DiscreteMixin


class Survey(DiscreteMixin, BaseExampleModel):
    """
    References
    ----------
    - :cite:p:`scutari_denis_2021`
    """

    _tags = {
        "name": "bnlearn/survey",
        "n_nodes": 6,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "discrete/survey.bif.gz"
