import os
import random
import unittest

import numpy as np
import pandas as pd
import pytest
from tqdm.auto import tqdm

from pgmpy.models import FunctionalBayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.utils import (
    discretize,
    get_example_model,
    llm_pairwise_orient,
    preprocess_data,
)


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


class TestPreprocessData(unittest.TestCase):
    def setUp(self):
        self.data_raw = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0
        )

        self.data_proc = self.data_raw.copy()
        self.data_proc["A_cat"] = self.data_proc.A_cat.astype("category")
        self.data_proc["B_cat"] = self.data_proc.C_cat.astype("category")
        self.data_proc["C_cat"] = self.data_proc.C_cat.astype("category")

        self.data_proc_proc = self.data_proc.copy()
        cat_type = pd.CategoricalDtype(
            categories=np.array(sorted(self.data_proc_proc.B_int.unique())),
            ordered=True,
        )

        self.data_proc_proc["B_int"] = self.data_proc_proc.B_int.astype(cat_type)

    def test_preprocess_data(self):
        df, dtypes = preprocess_data(self.data_raw)
        self.assertEqual(
            dtypes,
            {
                "A": "N",
                "B": "N",
                "C": "N",
                "A_cat": "C",
                "B_cat": "C",
                "C_cat": "C",
                "B_int": "N",
            },
        )

        df, dtypes = preprocess_data(self.data_proc)
        self.assertEqual(
            dtypes,
            {
                "A": "N",
                "B": "N",
                "C": "N",
                "A_cat": "C",
                "B_cat": "C",
                "C_cat": "C",
                "B_int": "N",
            },
        )

        df, dtypes = preprocess_data(self.data_proc_proc)
        self.assertEqual(
            dtypes,
            {
                "A": "N",
                "B": "N",
                "C": "N",
                "A_cat": "C",
                "B_cat": "C",
                "C_cat": "C",
                "B_int": "O",
            },
        )


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
        """Test loading of continuous Bayesian network models."""
        cont_models = {
            "ecoli70",
            "magic-niab",
            "magic-irri",
            "arth150",
            "sangiovese",
            "mehra",
        }

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

    def test_hybrid_model(self):
        """Test loading of hybrid Bayesian network models."""
        health = get_example_model("health")
        sangiovese = get_example_model("sangiovese")
        mehra = get_example_model("mehra")

        hybrid_modes = [health, sangiovese, mehra]
        for hybrid_model in hybrid_modes:
            self.assertIsNotNone(hybrid_model)
            self.assertTrue(hasattr(hybrid_model, "nodes"))
            self.assertTrue(hasattr(hybrid_model, "edges"))
            self.assertTrue(hasattr(hybrid_model, "get_cpds"))
            self.assertIsInstance(hybrid_model, FunctionalBayesianNetwork)

    def test_health_model_stimulate(self):
        """Verify that simulated samples from the health model have expected properties."""
        health = get_example_model("health")
        samples = health.simulate(n_samples=200, seed=42)
        self.assertTrue(set(samples["A"].unique()).issubset({"young", "adult", "old"}))
        self.assertTrue(set(samples["C"].unique()).issubset({"none", "mild", "severe"}))
        self.assertTrue(set(samples["H"].unique()).issubset({"none", "any"}))

        self.assertTrue(samples["D"].min() >= -0.5)

        young_samples = samples[samples["A"] == "young"]
        adult_samples = samples[samples["A"] == "adult"]
        old_samples = samples[samples["A"] == "old"]

        if not young_samples.empty:
            self.assertAlmostEqual(young_samples["O"].mean(), 60, delta=10)
        if not adult_samples.empty:
            self.assertAlmostEqual(adult_samples["O"].mean(), 180, delta=10)
        if not old_samples.empty:
            self.assertAlmostEqual(old_samples["O"].mean(), 360, delta=10)

        i_correlation = samples["I"].corr(samples["T"])
        o_correlation = samples["O"].corr(samples["T"])
        self.assertGreater(i_correlation, 0.5)
        self.assertGreater(o_correlation, 0.5)

        a_counts = samples["A"].value_counts(normalize=True)
        self.assertAlmostEqual(a_counts.get("young", 0), 0.35, delta=0.05)
        self.assertAlmostEqual(a_counts.get("adult", 0), 0.45, delta=0.05)
        self.assertAlmostEqual(a_counts.get("old", 0), 0.2, delta=0.05)

    def test_sangiovese_model_stimulate(self):
        """Verify that simulated samples from the sangiovese model have expected properties."""
        sangiovese = get_example_model("sangiovese")
        samples = sangiovese.simulate(n_samples=200, seed=42)
        self.assertTrue(
            set(samples["Treatment"].unique()).issubset(
                {
                    "T1a",
                    "T1b",
                    "T2a",
                    "T2b",
                    "T3a",
                    "T3b",
                    "T4a",
                    "T4b",
                    "T5a",
                    "T5b",
                    "T6a",
                    "T6b",
                    "T7a",
                    "T7b",
                    "T8a",
                    "T8b",
                }
            )
        )

        treatment_counts = samples["Treatment"].value_counts(normalize=True)
        self.assertAlmostEqual(treatment_counts.get("T1a", 0), 0.0592, delta=0.02)
        self.assertAlmostEqual(treatment_counts.get("T5b", 0), 0.0653, delta=0.02)
        self.assertAlmostEqual(treatment_counts.get("T8b", 0), 0.0637, delta=0.02)

        sproutn_ndvi06_corr = samples["SproutN"].corr(samples["NDVI06"])
        spad06_ndvi06_corr = samples["SPAD06"].corr(samples["NDVI06"])

        self.assertGreater(sproutn_ndvi06_corr, 0)
        self.assertGreater(spad06_ndvi06_corr, 0)

        bunchn_anthoc_corr = samples["BunchN"].corr(samples["Anthoc"])
        woodw_anthoc_corr = samples["WoodW"].corr(samples["Anthoc"])
        ndvi08_anthoc_corr = samples["NDVI08"].corr(samples["Anthoc"])

        self.assertLess(bunchn_anthoc_corr, 0)
        self.assertLess(woodw_anthoc_corr, 0)
        self.assertLess(ndvi08_anthoc_corr, 0)

        anthoc_polyph_corr = samples["Anthoc"].corr(samples["Polyph"])
        brix_polyph_corr = samples["Brix"].corr(samples["Polyph"])

        self.assertGreater(anthoc_polyph_corr, 0)
        self.assertGreater(brix_polyph_corr, 0)

    def test_mehra_model_simulation(self):
        """Verify that simulated samples from the mehra model have expected properties."""

        mehra = get_example_model("mehra")
        samples = mehra.simulate(n_samples=200, seed=42)
        self.assertTrue(
            set(samples["Region"].unique()).issubset(
                set(
                    [
                        "East Midlands",
                        "East of England",
                        "Greater London Authority",
                        "North East",
                        "North West",
                        "South East",
                        "South West",
                        "West Midlands",
                        "Yorkshire and The Humber",
                    ]
                )
            )
        )

        self.assertTrue(
            set(samples["Type"].unique()).issubset(
                set(
                    [
                        "Background Rural",
                        "Background Suburban",
                        "Background Urban",
                        "Industrial Suburban",
                        "Industrial Urban",
                        "Traffic Urban",
                    ]
                )
            )
        )

        region_counts = samples["Region"].value_counts(normalize=True)
        type_counts = samples["Type"].value_counts(normalize=True)

        self.assertAlmostEqual(
            region_counts.get("Greater London Authority", 0), 0.2161, delta=0.03
        )
        self.assertAlmostEqual(type_counts.get("Background Urban", 0), 0.5, delta=0.05)
        self.assertAlmostEqual(type_counts.get("Traffic Urban", 0), 0.3025, delta=0.04)

        self.assertAlmostEqual(samples["Latitude"].mean(), 52.4435, delta=2.0)
        self.assertAlmostEqual(samples["Longitude"].mean(), -1.1804, delta=2.0)

        urban_traffic = samples[samples["Type"] == "Traffic Urban"]
        rural_background = samples[samples["Type"] == "Background Rural"]

        if not urban_traffic.empty and not rural_background.empty:
            self.assertGreater(
                urban_traffic["no2"].mean(), rural_background["no2"].mean()
            )

        lat_t2m_corr = samples["Latitude"].corr(samples["t2m"])
        long_ws_corr = samples["Longitude"].corr(samples["ws"])

        self.assertIsNotNone(lat_t2m_corr)
        self.assertIsNotNone(long_ws_corr)

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
