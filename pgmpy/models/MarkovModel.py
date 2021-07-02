import warnings

from pgmpy.models import MarkovNetwork


class MarkovModel(MarkovNetwork):
    def __init__(self, ebunch=None, latents=set()):
        warnings.warn(
            "MarkovModel has been renamed to MarkovNetwork. MarkovModel will be removed in v0.1.17.",
            FutureWarning,
        )
        super(MarkovModel, self).__init__(ebunch=ebunch, latents=latents)
