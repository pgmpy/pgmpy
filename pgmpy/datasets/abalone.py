from pgmpy.datasets import dataset_class
from pgmpy.datasets._base import _BaseDataset


@dataset_class
class Abalone(_BaseDataset):
    name = "abalone"
    tags = {
        "has_ground_truth": True,
        "is_simulated": False,
        "n_variables": 9,
        "n_samples": 4177,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/abalone/"

    default_variant = "mixed_numeric"
    variant_urls = {
        "continuous": base_url + "data/abalone.continuous.txt",
        "mixed_numeric": base_url + "data/abalone.mixed.numeric.txt",
        "mixed_max3": base_url + "data/abalone.mixed.maximum.3.txt",
    }
    ground_truth_url = base_url + "ground.truth/abalone.knowledge.txt"
