class BayesianNetwork(object):
    def __init__(self, ebunch=None, latents=set(), lavaan_str=None, dagitty_str=None):
        raise ImportError(
            "BayesianNetwork has been deprecated. Please use DiscreteBayesianNetwork instead."
        )
