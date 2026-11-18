import random
import unittest

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.utils import (
    discretize,
    get_example_model,
)
from pgmpy.utils.mathext import sample_discrete


class TestDiscretization(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal(1000)
        Y = 0.2 * X + rng.standard_normal(1000)
        Z = 0.4 * X + 0.5 * Y + rng.standard_normal(1000)

        self.data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_rounding_disc(self):
        df_disc = discretize(data=self.data, cardinality={"X": 5, "Y": 4, "Z": 3}, method="rounding")
        self.assertEqual(df_disc["X"].nunique(), 5)
        self.assertEqual(df_disc["Y"].nunique(), 4)
        self.assertEqual(df_disc["Z"].nunique(), 3)

        df_disc = discretize(data=self.data, cardinality={"X": 5, "Y": 4, "Z": 3}, method="quantile")
        self.assertEqual(df_disc["X"].nunique(), 5)
        self.assertEqual(df_disc["Y"].nunique(), 4)
        self.assertEqual(df_disc["Z"].nunique(), 3)


class TestGetExampleModel(unittest.TestCase):
    def test_get_categorical_models(self):
        """Test loading of categorical Bayesian network models."""
        cat_models = {
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
        }

        # Randomly select 5 categorical models to test
        choices = random.sample(list(cat_models), k=5)
        for model in tqdm(choices, desc="Testing categorical models"):
            m = get_example_model(model=model)
            # Basic model validation
            self.assertIsNotNone(m)
            self.assertTrue(hasattr(m, "nodes"))
            self.assertTrue(hasattr(m, "edges"))
            del m

    def test_get_continuous_models(self):
        # Test ecoli70 model specifically as we have its structure
        model = get_example_model("ecoli70")
        self.assertIsInstance(model, LinearGaussianBayesianNetwork)
        self.assertEqual(len(model.nodes()), 46)  # Number of nodes in ecoli70

        # Verify some known relationships from the provided structure
        self.assertIn(("asnA", "icdA"), model.edges())
        self.assertIn(("asnA", "lacA"), model.edges())
        self.assertIn(("sucA", "atpD"), model.edges())

        # Verify CPD structure for a known node
        cpd = model.get_cpds("aceB")
        self.assertIsNotNone(cpd)
        self.assertEqual(cpd.variable, "aceB")
        self.assertEqual(len(cpd.evidence), 1)
        self.assertIn("icdA", cpd.evidence)

    def test_get_example_model_dagitty(self):
        dag_models = [
            "M-bias",
            "confounding",
            "mediator",
            "paths",
            "Sebastiani_2005",
            "Polzer_2012",
            "Schipf_2010",
            "Shrier_2008",
            "Acid_1996",
            "Thoemmes_2013",
            "Kampen_2014",
            "Didelez_2010",
        ]
        # Would take too much time to load all the models. Hence, randomly select
        # 3 and try to load them.
        choices = random.sample(dag_models, k=3)
        for model in tqdm(choices):
            print(model)
            m = get_example_model(model=model)
            self.assertIsNotNone(m)
            self.assertTrue(hasattr(m, "nodes"))
            self.assertTrue(hasattr(m, "edges"))
            del m

    def test_invalid_model_name(self):
        """Test handling of invalid model names."""
        with self.assertRaises(ValueError):
            get_example_model("nonexistent_model")

    def test_model_categorization(self):
        """Test that all models are properly categorized."""
        # Test a model from each category
        cat_model = get_example_model("asia")
        self.assertNotIsInstance(cat_model, LinearGaussianBayesianNetwork)

        cont_model = get_example_model("magic-irri")
        self.assertIsInstance(cont_model, LinearGaussianBayesianNetwork)


class TestSampleDiscrete(unittest.TestCase):
    def test_sample_discrete_with_and_without_seed(self):
        values = np.array(["a", "b", "c"])
        weights = np.array([0.2, 0.5, 0.3])

        # Check reproducible values with seed
        result1 = sample_discrete(values, weights, 50, rng=0)
        result2 = sample_discrete(values, weights, 50, rng=0)
        np.testing.assert_array_equal(result1, result2)

        self.assertTrue(set(result1.tolist()).issubset({"a", "b", "c"}))
