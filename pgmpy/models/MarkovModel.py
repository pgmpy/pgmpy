from pgmpy.global_vars import logger
from pgmpy.models import MarkovNetwork


class MarkovModel(MarkovNetwork):
    def __init__(self, ebunch=None, latents=set()):
        logger.warning(
            "MarkovModel has been renamed to MarkovNetwork. Please use MarkovNetwork class, MarkovModel will be removed in future."
        )
        super(MarkovModel, self).__init__(ebunch=ebunch, latents=latents)
