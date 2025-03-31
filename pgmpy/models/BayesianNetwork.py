from pgmpy.global_vars import logger
from pgmpy.models import DiscreteBayesianNetwork


class BayesianNetwork(DiscreteBayesianNetwork):
    def __init__(self, ebunch=None, latents=set(), lavaan_str=None, dagitty_str=None):
        logger.warning(
            "BayesianNetwork class is deprecated. Please use DiscreteBayesianNetwork class instead."
        )
        super(BayesianNetwork, self).__init__(ebunch, latents, lavaan_str, dagitty_str)
