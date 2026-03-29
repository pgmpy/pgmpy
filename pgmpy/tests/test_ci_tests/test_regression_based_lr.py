import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import statsmodels.tools.sm_exceptions as sm_exceptions
from skbase.lookup import all_objects

from pgmpy.ci_tests import RegressionBasedLR, _BaseCITest


class TestRegressionBasedLR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2024)
        n = 1000

        # Fork: X <- Z -> Y  (X ind Y | Z, X dep Y marginally)
        Z = rng.standard_normal(n)
        X_cont = 2.0 * Z + rng.standard_normal(n)
        Y_cont = 3.0 * Z + rng.standard_normal(n)
        cls.cont_data = pd.DataFrame({"X": X_cont, "Y": Y_cont, "Z": Z})

        # Binary X: X is Bernoulli(sigmoid(Z)), Y continuous
        rng2 = np.random.default_rng(7)
        Z2 = rng2.standard_normal(n)
        prob = 1 / (1 + np.exp(-Z2))
        X_bin = (rng2.uniform(size=n) < prob).astype(int)
        Y_bin = 2.0 * Z2 + rng2.standard_normal(n)
        cls.binary_data = pd.DataFrame({"X": X_bin.astype(str), "Y": Y_bin, "Z": Z2})

        # Multinomial X (3 classes) dependent on Z
        rng3 = np.random.default_rng(13)
        Z3 = rng3.standard_normal(n)
        X_multi = pd.cut(
            Z3 + rng3.standard_normal(n) * 0.3,
            bins=3,
            labels=["low", "mid", "high"],
        )
        Y_multi = 2.0 * Z3 + rng3.standard_normal(n)
        cls.multi_data = pd.DataFrame({"X": X_multi, "Y": Y_multi, "Z": Z3})

    # ------------------------------------------------------------------
    # Continuous X tests
    # ------------------------------------------------------------------

    def test_continuous_conditional_independence(self):
        """X ind Y | Z in fork X <- Z -> Y."""
        test = RegressionBasedLR(data=self.cont_data)
        result = test("X", "Y", ["Z"], significance_level=0.05)
        self.assertTrue(result)
        self.assertGreaterEqual(test.p_value_, 0.05)
        self.assertIsInstance(test.statistic_, float)
        self.assertIsInstance(test.dof_, int)

    def test_continuous_marginal_dependence(self):
        """X dep Y marginally (no conditioning) in fork."""
        test = RegressionBasedLR(data=self.cont_data)
        result = test("X", "Y", [], significance_level=0.05)
        self.assertFalse(result)
        self.assertLess(test.p_value_, 0.05)

    # ------------------------------------------------------------------
    # Binary X test
    # ------------------------------------------------------------------

    def test_binary_marginal_dependence(self):
        """Binary X dep Y marginally; exercises binary Logit path."""
        test = RegressionBasedLR(data=self.binary_data)
        result = test("X", "Y", [], significance_level=0.05)
        self.assertFalse(result)
        self.assertLess(test.p_value_, 0.05)

    # ------------------------------------------------------------------
    # Multinomial X test
    # ------------------------------------------------------------------

    def test_multinomial_conditional_independence(self):
        """Multinomial X ind Y | Z; exercises MNLogit path and dof calculation."""
        test = RegressionBasedLR(data=self.multi_data)
        result = test("X", "Y", ["Z"], significance_level=0.05)
        self.assertTrue(result)
        self.assertGreaterEqual(test.p_value_, 0.05)
        self.assertGreater(test.dof_, 0)

    # ------------------------------------------------------------------
    # _encode_features: categorical predictor branch (Group 1 coverage)
    # ------------------------------------------------------------------

    def test_categorical_y_as_predictor(self):
        """Categorical Y is one-hot encoded in _encode_features (get_dummies branch)."""
        rng = np.random.default_rng(101)
        n = 300
        Y_cat = pd.Categorical(rng.choice(["low", "mid", "high"], size=n))
        Z = rng.standard_normal(n)
        X = Z + rng.standard_normal(n)
        data = pd.DataFrame({"X": X, "Y": Y_cat, "Z": Z})
        test = RegressionBasedLR(data=data)
        test.run_test("X", "Y", ["Z"])
        self.assertIsNotNone(test.p_value_)
        self.assertIsInstance(test.statistic_, float)

    # ------------------------------------------------------------------
    # Exception handler coverage (Group 2): mock raises actual errors
    # ------------------------------------------------------------------

    def test_logit_linalg_error_returns_independence(self):
        """LinAlgError during Logit.fit is caught; returns independence."""
        rng = np.random.default_rng(42)
        n = 200
        data = pd.DataFrame(
            {
                "X": pd.Categorical(rng.choice(["A", "B"], size=n)),
                "Y": rng.standard_normal(n),
            }
        )
        test = RegressionBasedLR(data=data)
        with patch(
            "statsmodels.discrete.discrete_model.Logit.fit",
            side_effect=np.linalg.LinAlgError("Singular matrix"),
        ):
            test.run_test("X", "Y", [])
        self.assertEqual(test.statistic_, 0.0)
        self.assertEqual(test.p_value_, 1.0)
        self.assertEqual(test.dof_, 0)

    def test_mnlogit_perfect_separation_error_returns_independence(self):
        """PerfectSeparationError during MNLogit.fit is caught; returns independence."""
        rng = np.random.default_rng(42)
        n = 200
        data = pd.DataFrame(
            {
                "X": pd.Categorical(rng.choice(["A", "B", "C"], size=n)),
                "Y": rng.standard_normal(n),
            }
        )
        test = RegressionBasedLR(data=data)
        with patch(
            "statsmodels.discrete.discrete_model.MNLogit.fit",
            side_effect=sm_exceptions.PerfectSeparationError("Perfect separation"),
        ):
            test.run_test("X", "Y", [])
        self.assertEqual(test.statistic_, 0.0)
        self.assertEqual(test.p_value_, 1.0)
        self.assertEqual(test.dof_, 0)

    # ------------------------------------------------------------------
    # Degenerate OLS df2 <= 0 branch (Group 4 coverage)
    # ------------------------------------------------------------------

    def test_ols_degenerate_dof(self):
        """n=3 with 2 Z cols gives df2 = 3 - 4 = -1 <= 0; returns independence."""
        rng = np.random.default_rng(77)
        n = 3
        data = pd.DataFrame(
            {
                "X": rng.standard_normal(n),
                "Y": rng.standard_normal(n),
                "Z1": rng.standard_normal(n),
                "Z2": rng.standard_normal(n),
            }
        )
        test = RegressionBasedLR(data=data)
        test.run_test("X", "Y", ["Z1", "Z2"])
        self.assertEqual(test.p_value_, 1.0)
        self.assertEqual(test.statistic_, 0.0)

    # ------------------------------------------------------------------
    # Edge cases / defensive branches
    # ------------------------------------------------------------------

    def test_constant_variable(self):
        """Constant X (zero variance) does not crash; returns independence."""
        rng = np.random.default_rng(7)
        data = pd.DataFrame({"X": [1.0] * 200, "Y": rng.standard_normal(200)})
        test = RegressionBasedLR(data=data)
        result = test("X", "Y", [], significance_level=0.05)
        self.assertTrue(result)

    def test_constant_categorical_single_class(self):
        """Constant categorical X (1 class) hits n_classes < 2 early return."""
        rng = np.random.default_rng(123)
        n = 100
        data = pd.DataFrame({"X": ["A"] * n, "Y": rng.standard_normal(n)})
        test = RegressionBasedLR(data=data)
        result = test("X", "Y", [], significance_level=0.05)
        self.assertTrue(result)
        self.assertEqual(test.statistic_, 0.0)
        self.assertEqual(test.p_value_, 1.0)
        self.assertEqual(test.dof_, 0)

    def test_rank_deficient_design_matrix(self):
        """Rank-deficient predictors emit a warning but do not crash."""
        rng = np.random.default_rng(55)
        n = 200
        Z = rng.standard_normal(n)
        data = pd.DataFrame(
            {
                "X": rng.standard_normal(n),
                "Y": Z,
                "Z": Z,  # Z == Y → collinear columns in design matrix
            }
        )
        test = RegressionBasedLR(data=data)
        with self.assertLogs("pgmpy", level="WARNING"):
            test.run_test("X", "Y", ["Z"])
        self.assertIsNotNone(test.p_value_)

    def test_all_nan_raises(self):
        """All-NaN data after dropna raises ValueError (n == 0 guard)."""
        data = pd.DataFrame({"X": [np.nan] * 10, "Y": [np.nan] * 10})
        test = RegressionBasedLR(data=data)
        with self.assertRaises(ValueError):
            test.run_test("X", "Y", [])

    # ------------------------------------------------------------------
    # Input validation (from _BaseCITest)
    # ------------------------------------------------------------------

    def test_x_equals_y_raises(self):
        """X == Y must raise ValueError via _BaseCITest._validate_inputs."""
        test = RegressionBasedLR(data=self.cont_data)
        with self.assertRaises(ValueError):
            test("X", "X", [])

    def test_x_in_z_raises(self):
        """X in Z must raise ValueError via _BaseCITest._validate_inputs."""
        test = RegressionBasedLR(data=self.cont_data)
        with self.assertRaises(ValueError):
            test("X", "Y", ["X"])

    # ------------------------------------------------------------------
    # Registry check
    # ------------------------------------------------------------------

    def test_in_ci_registry(self):
        """RegressionBasedLR is discoverable via skbase all_objects."""
        all_names = [
            cls.get_class_tag("name")
            for cls in all_objects(
                object_types=_BaseCITest,
                package_name="pgmpy.ci_tests",
                return_names=False,
            )
        ]
        self.assertIn("regression_based_lr", all_names)


class TestRegressionBasedLRIntegration(unittest.TestCase):
    """Integration test: use RegressionBasedLR as CI test in PC algorithm."""

    def test_pc_recovers_fork_skeleton(self):
        """PC with RegressionBasedLR correctly recovers a fork skeleton."""
        from pgmpy.causal_discovery import PC

        rng = np.random.default_rng(42)
        n = 500
        Z = rng.standard_normal(n)
        X = 2.0 * Z + rng.standard_normal(n)
        Y = 3.0 * Z + rng.standard_normal(n)
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        est = PC(ci_test="regression_based_lr", significance_level=0.01)
        est.fit(data)
        edges = set(est.skeleton_.edges())
        sym_edges = edges | {(v, u) for u, v in edges}
        self.assertIn(("X", "Z"), sym_edges)
        self.assertIn(("Y", "Z"), sym_edges)
        self.assertNotIn(("X", "Y"), sym_edges)
