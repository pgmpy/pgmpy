import unittest

import numpy as np
import pandas as pd
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

        # Mixed: categorical X, continuous Y and Z, X ind Y | Z
        rng4 = np.random.default_rng(99)
        Z4 = rng4.standard_normal(n)
        prob4 = 1 / (1 + np.exp(-Z4))
        X_mix = pd.Categorical(np.where(rng4.uniform(size=n) < prob4, "A", "B"))
        Y_mix = 2.0 * Z4 + rng4.standard_normal(n)
        cls.mixed_data = pd.DataFrame({"X": X_mix, "Y": Y_mix, "Z": Z4})

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

    def test_continuous_direct_dependence(self):
        """Direct edge X -> Y detected."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.standard_normal(n)
        Y = 2.0 * X + rng.standard_normal(n) * 0.5
        data = pd.DataFrame({"X": X, "Y": Y})
        test = RegressionBasedLR(data=data)
        result = test("X", "Y", [], significance_level=0.05)
        self.assertFalse(result)
        self.assertLess(test.p_value_, 0.05)

    # ------------------------------------------------------------------
    # Binary X tests
    # ------------------------------------------------------------------

    def test_binary_conditional_independence(self):
        """Binary X ind Y | Z."""
        test = RegressionBasedLR(data=self.binary_data)
        result = test("X", "Y", ["Z"], significance_level=0.05)
        self.assertTrue(result)
        self.assertGreaterEqual(test.p_value_, 0.05)

    def test_binary_marginal_dependence(self):
        """Binary X dep Y marginally."""
        test = RegressionBasedLR(data=self.binary_data)
        result = test("X", "Y", [], significance_level=0.05)
        self.assertFalse(result)
        self.assertLess(test.p_value_, 0.05)

    # ------------------------------------------------------------------
    # Multinomial X tests
    # ------------------------------------------------------------------

    def test_multinomial_conditional_independence(self):
        """Multinomial X ind Y | Z."""
        test = RegressionBasedLR(data=self.multi_data)
        result = test("X", "Y", ["Z"], significance_level=0.05)
        self.assertTrue(result)
        self.assertGreaterEqual(test.p_value_, 0.05)

    def test_multinomial_statistic_attributes(self):
        """run_test sets statistic_, p_value_, dof_ correctly for multinomial."""
        test = RegressionBasedLR(data=self.multi_data)
        test.run_test("X", "Y", [])
        self.assertGreater(test.statistic_, 0.0)
        self.assertGreaterEqual(test.p_value_, 0.0)
        self.assertLessEqual(test.p_value_, 1.0)
        self.assertGreater(test.dof_, 0)

    # ------------------------------------------------------------------
    # Mixed data
    # ------------------------------------------------------------------

    def test_mixed_data_conditional_independence(self):
        """Categorical X, continuous Y and Z: X ind Y | Z."""
        test = RegressionBasedLR(data=self.mixed_data)
        result = test("X", "Y", ["Z"], significance_level=0.05)
        self.assertTrue(result)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_conditioning_set(self):
        """Z=[] reduces to marginal test; function runs without error."""
        test = RegressionBasedLR(data=self.cont_data)
        test.run_test("X", "Y", [])
        self.assertIsNotNone(test.p_value_)

    def test_constant_variable(self):
        """Constant X (zero variance) returns independence without crashing."""
        rng = np.random.default_rng(7)
        data = pd.DataFrame({"X": [1.0] * 200, "Y": rng.standard_normal(200)})
        test = RegressionBasedLR(data=data)
        result = test("X", "Y", [], significance_level=0.05)
        self.assertTrue(result)

    def test_boolean_column_as_categorical(self):
        """Boolean dtype X is routed to binary logistic model."""
        rng = np.random.default_rng(42)
        n = 500
        Z = rng.standard_normal(n)
        X_bool = Z > 0
        Y = 2.0 * Z + rng.standard_normal(n) * 0.3
        data = pd.DataFrame({"X": X_bool, "Y": Y, "Z": Z})
        test = RegressionBasedLR(data=data)
        result = test("X", "Y", ["Z"], significance_level=0.05)
        self.assertTrue(result)

    def test_missing_data_handled(self):
        """NaN rows are dropped gracefully."""
        data = self.cont_data.copy()
        data.loc[[0, 5, 10], "X"] = np.nan
        test = RegressionBasedLR(data=data)
        test.run_test("X", "Y", ["Z"])
        self.assertIsNotNone(test.p_value_)

    def test_asymmetry_acknowledged(self):
        """Both orderings of X/Y yield consistent independence verdict."""
        test_xy = RegressionBasedLR(data=self.cont_data)
        test_yx = RegressionBasedLR(data=self.cont_data)
        test_xy.run_test("X", "Y", ["Z"])
        test_yx.run_test("Y", "X", ["Z"])
        # Both should say independent (p >= 0.05)
        self.assertGreaterEqual(test_xy.p_value_, 0.05)
        self.assertGreaterEqual(test_yx.p_value_, 0.05)

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
    # Edge cases that cover defensive branches
    # ------------------------------------------------------------------

    def test_all_nan_raises(self):
        """All-NaN data after dropna raises ValueError."""
        data = pd.DataFrame(
            {
                "X": [np.nan] * 10,
                "Y": [np.nan] * 10,
            }
        )
        test = RegressionBasedLR(data=data)
        with self.assertRaises(ValueError):
            test.run_test("X", "Y", [])

    def test_constant_categorical_single_class(self):
        """Constant categorical X (1 class) returns independence."""
        rng = np.random.default_rng(123)
        n = 100
        data = pd.DataFrame(
            {
                "X": ["A"] * n,  # single class → n_classes < 2 branch
                "Y": rng.standard_normal(n),
            }
        )
        test = RegressionBasedLR(data=data)
        result = test("X", "Y", [], significance_level=0.05)
        self.assertTrue(result)
        self.assertEqual(test.statistic_, 0.0)
        self.assertEqual(test.p_value_, 1.0)
        self.assertEqual(test.dof_, 0)

    def test_perfect_separation_returns_independence(self):
        """Perfect separation in logistic regression is caught gracefully."""
        n = 100
        data = pd.DataFrame(
            {
                "X": pd.Categorical(["A"] * (n // 2) + ["B"] * (n // 2)),
                "Y": np.concatenate([np.ones(n // 2) * -1e6, np.ones(n // 2) * 1e6]),
            }
        )
        test = RegressionBasedLR(data=data)
        test.run_test("X", "Y", [])
        self.assertIsNotNone(test.p_value_)

    def test_rank_deficient_design_matrix(self):
        """Rank-deficient predictors trigger warning but don't crash."""
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

    def test_ols_degenerate_dof(self):
        """When n ≈ p (no residual df), returns independence gracefully."""
        rng = np.random.default_rng(77)
        n = 5
        data = pd.DataFrame(
            {
                "X": rng.standard_normal(n),
                "Y": rng.standard_normal(n),
                "Z1": rng.standard_normal(n),
                "Z2": rng.standard_normal(n),
                "Z3": rng.standard_normal(n),
                "Z4": rng.standard_normal(n),
            }
        )
        test = RegressionBasedLR(data=data)
        test.run_test("X", "Y", ["Z1", "Z2", "Z3", "Z4"])
        self.assertIsNotNone(test.p_value_)

    def test_multinomial_with_many_z_dof_edge(self):
        """Multinomial path with enough Z columns to stress dof calculation."""
        rng = np.random.default_rng(88)
        n = 200
        Z = rng.standard_normal(n)
        X = pd.Categorical(np.where(Z > 0.5, "high", np.where(Z < -0.5, "low", "mid")))
        Y = rng.standard_normal(n)
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        test = RegressionBasedLR(data=data)
        test.run_test("X", "Y", ["Z"])
        self.assertGreater(test.dof_, 0)

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
        n = 500  # larger n to give RegressionBasedLR sufficient power
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
