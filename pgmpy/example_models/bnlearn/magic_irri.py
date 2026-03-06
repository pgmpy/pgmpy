from .._base import ContinuousMixin, _BaseExampleModel


class MagicIRRI(ContinuousMixin, _BaseExampleModel):
    """
    References
    ----------
    ..[1] Model developed as an example of multiple trait modelling in plant genetics for the invited
          talk "Bayesian Networks, MAGIC Populations and Multiple Trait Prediction" delivered by Marco
          Scutari at the 5th International Conference on Quantitative Genetics (ICQG 2016).
    """

    _tags = {
        "name": "bnlearn/magic_irri",
        "n_nodes": 64,
        "n_edges": 102,
        "is_parameterized": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_hybrid": False,
    }
    data_url = "continuous/magic-irri.json"
