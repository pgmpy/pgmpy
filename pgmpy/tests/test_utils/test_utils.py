import random
import unittest

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pgmpy.utils import discretize, get_example_model


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


class TestDiscretization(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal(1000)
        Y = 0.2 * X + rng.standard_normal(1000)
        Z = 0.4 * X + 0.5 * Y + rng.standard_normal(1000)

        self.data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_rounding_disc(self):
        df_disc = discretize(
            data=self.data, cardinality={"X": 5, "Y": 4, "Z": 3}, method="rounding"
        )
        self.assertEqual(df_disc["X"].nunique(), 5)
        self.assertEqual(df_disc["Y"].nunique(), 4)
        self.assertEqual(df_disc["Z"].nunique(), 3)

        df_disc = discretize(
            data=self.data, cardinality={"X": 5, "Y": 4, "Z": 3}, method="quantile"
        )
        self.assertEqual(df_disc["X"].nunique(), 5)
        self.assertEqual(df_disc["Y"].nunique(), 4)
        self.assertEqual(df_disc["Z"].nunique(), 3)
