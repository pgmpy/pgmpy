import unittest
import random

from tqdm.auto import tqdm

from pgmpy.utils import get_example_model


class TestDAGCreation(unittest.TestCase):
    def test_get_example_model(self):
        all_models = [
            "asia",
            "cancer",
            "earthquake",
            "sachs",
            "survey",
            "alarm",
            "barley",
            "child",
            "insurance",
            "mildew",
            "water",
            "hailfinder",
            "hepar2",
            "win95pts",
            "andes",
            "diabetes",
            "link",
            "munin1",
            "munin2",
            "munin3",
            "munin4",
            "pathfinder",
            "pigs",
            "munin",
        ]
        # Would take too much time to load all the models. Hence, randomly select
        # 5 and try to load them.
        choices = random.choices(all_models, k=5)
        for model in tqdm(choices):
            m = get_example_model(model=model)
            del m
