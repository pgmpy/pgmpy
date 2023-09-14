from pgmpy.global_vars import logger
from pgmpy.models import BayesianNetwork


class BayesianModel(BayesianNetwork):
    def __init__(self, ebunch=None, latents=set()):
        logger.warning(
            "BayesianModel has been renamed to BayesianNetwork. Please use BayesianNetwork class, BayesianModel will be removed in future."
        )
        super(BayesianModel, self).__init__(ebunch=ebunch, latents=latents)
