from pgmpy.global_vars import logger


class BayesianNetwork(object):
    def __init__(self, ebunch=None, latents=set(), lavaan_str=None, dagitty_str=None):
        logger.error(
            "BayesianNetwork class is deprecated. Please use DiscreteBayesianNetwork class instead."
        )
