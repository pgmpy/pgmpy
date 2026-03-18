from pgmpy.models import DiscreteMarkovNetwork


class MarkovNetwork(DiscreteMarkovNetwork):
    def __init__(self, *args, **kwargs):
        print(
            "WARNING: MarkovNetwork has been deprecated. Please use DiscreteMarkovNetwork instead."
        )
        super().__init__(*args, **kwargs)
