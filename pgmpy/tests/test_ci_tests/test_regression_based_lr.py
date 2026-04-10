import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from skbase.lookup import all_objects

from pgmpy.ci_tests import RegressionBasedLR, _BaseCITest
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


def _fork_data(seed=42):
    """Return data simulated from a fork Z -> X, Z -> Y (X _|_ Y | Z)."""
    model = LinearGaussianBayesianNetwork([("Z", "X"), ("Z", "Y")])
    model.add_cpds(
        LinearGaussianCPD("Z", [0], 1),
        LinearGaussianCPD("X", [0, 2], 1, ["Z"]),
        LinearGaussianCPD("Y", [0, 3], 1, ["Z"]),
    )
    return model.simulate(n_samples=1000, seed=seed)


class TestRegressionBasedLR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cont_data = _fork_data()

    # ------------------------------------------------------------------
    # Core paths: conditional independence and marginal dependence
    # ------------------------------------------------------------------

    def test_continuous_ci_and_marginal_dep(self):
        """X _|_ Y | Z in fork; X dep Y marginally."""
        test = RegressionBasedLR(self.cont_data)
        self.assertTrue(test("X", "Y", ["Z"], significance_level=0.05))
        self.assertGreaterEqual(test.p_value_, 0.05)
        self.assertIsInstance(test.statistic_, float)
        self.assertIsInstance(test.dof_, int)

        self.assertFalse(test("X", "Y", [], significance_level=0.05))
        self.assertLess(test.p_value_, 0.05)

    def test_categorical_x_marginal_dependence(self):
        """Binary and multinomial X are each dependent on Z marginally."""
        rng = np.random.default_rng(7)
        n = 800
        Z = rng.standard_normal(n)
        prob = 1 / (1 + np.exp(-Z))
        X_bin = (rng.uniform(size=n) < prob).astype(str)
        X_multi = pd.cut(Z + rng.standard_normal(n) * 0.3, bins=3, labels=["low", "mid", "high"])
        data = pd.DataFrame({"X_bin": X_bin, "X_multi": X_multi, "Z": Z})

        for x_col, family in [("X_bin", "logistic"), ("X_multi", "multinomial")]:
            with self.subTest(x_col=x_col, family=family):
                test = RegressionBasedLR(data, regression_family=family)
                result = test(x_col, "Z", [], significance_level=0.05)
                self.assertFalse(result)
                self.assertLess(test.p_value_, 0.05)

    # ------------------------------------------------------------------
    # regression_family constructor argument
    # ------------------------------------------------------------------

    def test_explicit_linear_family(self):
        """regression_family='linear' uses OLS regardless of column dtype."""
        test = RegressionBasedLR(self.cont_data, regression_family="linear")
        test.run_test("X", "Y", ["Z"])
        self.assertIsInstance(test.statistic_, float)

    def test_invalid_family_raises(self):
        with self.assertRaises(ValueError):
            RegressionBasedLR(self.cont_data, regression_family="poisson")

    def test_ordinal_family_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            RegressionBasedLR(self.cont_data, regression_family="ordinal")

    # ------------------------------------------------------------------
    # Degenerate inputs and error handling
    # ------------------------------------------------------------------

    def test_degenerate_inputs(self):
        """Constant X (continuous or 1-class categorical) returns independence."""
        rng = np.random.default_rng(7)
        n = 200
        Y = rng.standard_normal(n)

        for x_vals in ([1.0] * n, ["A"] * n):
            data = pd.DataFrame({"X": x_vals, "Y": Y})
            test = RegressionBasedLR(data)
            self.assertTrue(test("X", "Y", [], significance_level=0.05))
            self.assertEqual(test.p_value_, 1.0)

    def test_model_fitting_errors_return_independence(self):
        """LinAlgError or PerfectSeparationError during fit returns independence."""
        rng = np.random.default_rng(42)
        n = 200
        for x_vals, patch_target in [
            (
                pd.Categorical(rng.choice(["A", "B"], size=n)),
                "statsmodels.discrete.discrete_model.Logit.fit",
            ),
            (
                pd.Categorical(rng.choice(["A", "B", "C"], size=n)),
                "statsmodels.discrete.discrete_model.MNLogit.fit",
            ),
        ]:
            data = pd.DataFrame({"X": x_vals, "Y": rng.standard_normal(n)})
            test = RegressionBasedLR(data)
            with patch(patch_target, side_effect=np.linalg.LinAlgError("Singular matrix")):
                test.run_test("X", "Y", [])
            self.assertEqual(test.statistic_, 0.0)
            self.assertEqual(test.p_value_, 1.0)

    def test_all_nan_raises(self):
        """All-NaN data after dropna raises ValueError."""
        data = pd.DataFrame({"X": [np.nan] * 10, "Y": [np.nan] * 10})
        test = RegressionBasedLR(data)
        with self.assertRaises(ValueError):
            test.run_test("X", "Y", [])

    # ------------------------------------------------------------------
    # Input validation and registry
    # ------------------------------------------------------------------

    def test_input_validation(self):
        """X == Y or X in Z must raise ValueError."""
        test = RegressionBasedLR(self.cont_data)
        with self.assertRaises(ValueError):
            test("X", "X", [])
        with self.assertRaises(ValueError):
            test("X", "Y", ["X"])

    def test_in_ci_registry(self):
        """RegressionBasedLR is discoverable via skbase all_objects."""
        names = [
            cls.get_class_tag("name")
            for cls in all_objects(object_types=_BaseCITest, package_name="pgmpy.ci_tests", return_names=False)
        ]
        self.assertIn("regression_based_lr", names)


class TestRegressionBasedLRIntegration(unittest.TestCase):
    def test_pc_recovers_fork_skeleton(self):
        """PC with RegressionBasedLR correctly recovers a fork skeleton."""
        from pgmpy.causal_discovery import PC

        est = PC(ci_test="regression_based_lr", significance_level=0.01)
        est.fit(_fork_data(seed=0))
        edges = set(est.skeleton_.edges())
        sym_edges = edges | {(v, u) for u, v in edges}
        self.assertIn(("X", "Z"), sym_edges)
        self.assertIn(("Y", "Z"), sym_edges)
        self.assertNotIn(("X", "Y"), sym_edges)
