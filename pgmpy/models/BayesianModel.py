import warnings

from pgmpy.models import BayesianNetwork


class BayesianModel(BayesianNetwork):
    def __init__(self, ebunch=None, latents=set()):
        warnings.warn(
            "BayesianModel has been renamed to BayesianNetwork. BayesianModel will be removed in v0.1.17.",
            FutureWarning,
        )
        super(BayesianModel, self).__init__(ebunch=ebunch, latents=latents)
