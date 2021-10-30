import warnings

from pgmpy.models import BayesianNetwork


class BayesianModel(BayesianNetwork):
    def __init__(self, ebunch=None, latents=set()):
        warnings.warn(
            "BayesianModel has been renamed to BayesianNetwork. Please use BayesianNetwork class, BayesianModel will be removed in future.",
            FutureWarning,
        )
        super(BayesianModel, self).__init__(ebunch=ebunch, latents=latents)
