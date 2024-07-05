import os
import random
import unittest

import numpy as np
import pandas as pd
import pytest
from tqdm.auto import tqdm

from pgmpy.utils import discretize, get_example_model, llm_pairwise_orient


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


class TestPairwiseOrientation(unittest.TestCase):
    @pytest.mark.skipif(
        "GEMINI_API_KEY" not in os.environ, reason="Gemini API key is not set"
    )
    def test_llm(self):
        descriptions = {
            "Age": "The age of a person",
            "Workclass": "The workplace where the person is employed such as Private industry, or self employed",
            "Education": "The highest level of education the person has finished",
            "MaritalStatus": "The marital status of the person",
            "Occupation": "The kind of job the person does. For example, sales, craft repair, clerical",
            "Relationship": "The relationship status of the person",
            "Race": "The ethnicity of the person",
            "Sex": "The sex or gender of the person",
            "HoursPerWeek": "The number of hours per week the person works",
            "NativeCountry": "The native country of the person",
            "Income": "The income i.e. amount of money the person makes",
        }

        self.assertEqual(
            llm_pairwise_orient(
                x="Age", y="Income", descriptions=descriptions, domain="Social Sciences"
            ),
            ("Age", "Income"),
        )
        self.assertEqual(
            llm_pairwise_orient(
                x="Income", y="Age", descriptions=descriptions, domain="Social Sciences"
            ),
            ("Age", "Income"),
        )
