from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import DiscreteBayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.parameter_estimator import LinearGaussianMLE


@pytest.fixture
def synthetic_gaussian_data():
    """
    Centralized fixture generating synthetic linear Gaussian data from a known SCM:
      X1 = Normal(loc=2.0, scale=1.5)
      X2 = 0.5 + 1.8 * X1 + Normal(0, 0.8)
      X3 = -1.2 + 0.6 * X1 - 2.1 * X2 + Normal(0, 0.5)
      X4 = Normal(loc=-0.5, scale=2.0) [Independent root node]
    """
    rng = np.random.default_rng(42)
    n = 200

    x1 = rng.normal(loc=2.0, scale=1.5, size=n)
    x2 = 0.5 + 1.8 * x1 + rng.normal(loc=0.0, scale=0.8, size=n)
    x3 = -1.2 + 0.6 * x1 - 2.1 * x2 + rng.normal(loc=0.0, scale=0.5, size=n)
    x4 = rng.normal(loc=-0.5, scale=2.0, size=n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
    model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x1", "x3"), ("x2", "x3")])
    model.add_node("x4")
    return {"data": df, "model": model, "n_samples": n}


class TestLinearGaussianMLELifecycleAndEstimation:
    @pytest.mark.parametrize("std_estimator", ["unbiased", "mle"])
    def test_fit_parameters_against_ols_ground_truth(self, synthetic_gaussian_data, std_estimator):
        data = synthetic_gaussian_data["data"]
        model = synthetic_gaussian_data["model"]

        estimator = LinearGaussianMLE(std_estimator=std_estimator)
        fitted = estimator.fit(model, data)

        assert fitted is estimator
        assert hasattr(estimator, "parameters_")
        assert len(estimator.parameters_) == len(model.nodes())

        # 1. Root node x1
        cpd_x1 = next(cpd for cpd in estimator.parameters_ if cpd.variable == "x1")
        assert isinstance(cpd_x1, LinearGaussianCPD)
        assert cpd_x1.evidence == []
        expected_x1_mean = data["x1"].mean()
        ddof_root = 1 if std_estimator == "unbiased" else 0
        expected_x1_std = data["x1"].std(ddof=ddof_root)
        np.testing.assert_allclose(cpd_x1.beta, [expected_x1_mean], rtol=1e-5)
        np.testing.assert_allclose(cpd_x1.std, expected_x1_std, rtol=1e-5)

        # 2. Single parent node x2 (parent: x1)
        cpd_x2 = next(cpd for cpd in estimator.parameters_ if cpd.variable == "x2")
        assert cpd_x2.evidence == ["x1"]
        lr_x2 = LinearRegression().fit(data[["x1"]], data["x2"])
        residuals_x2 = data["x2"] - lr_x2.predict(data[["x1"]])
        ddof_x2 = (1 + 1) if std_estimator == "unbiased" else 0
        expected_x2_std = residuals_x2.std(ddof=ddof_x2)
        expected_x2_beta = np.append([lr_x2.intercept_], lr_x2.coef_)
        np.testing.assert_allclose(cpd_x2.beta, expected_x2_beta, rtol=1e-5)
        np.testing.assert_allclose(cpd_x2.std, expected_x2_std, rtol=1e-5)

        # 3. Multi parent node x3 (parents: x1, x2)
        cpd_x3 = next(cpd for cpd in estimator.parameters_ if cpd.variable == "x3")
        parents_x3 = ["x1", "x2"]
        assert cpd_x3.evidence == parents_x3
        lr_x3 = LinearRegression().fit(data[parents_x3], data["x3"])
        residuals_x3 = data["x3"] - lr_x3.predict(data[parents_x3])
        ddof_x3 = (1 + len(parents_x3)) if std_estimator == "unbiased" else 0
        expected_x3_std = residuals_x3.std(ddof=ddof_x3)
        expected_x3_beta = np.append([lr_x3.intercept_], lr_x3.coef_)
        np.testing.assert_allclose(cpd_x3.beta, expected_x3_beta, rtol=1e-5)
        np.testing.assert_allclose(cpd_x3.std, expected_x3_std, rtol=1e-5)

        # 4. Independent isolated root node x4
        cpd_x4 = next(cpd for cpd in estimator.parameters_ if cpd.variable == "x4")
        assert cpd_x4.evidence == []
        expected_x4_mean = data["x4"].mean()
        expected_x4_std = data["x4"].std(ddof=ddof_root)
        np.testing.assert_allclose(cpd_x4.beta, [expected_x4_mean], rtol=1e-5)
        np.testing.assert_allclose(cpd_x4.std, expected_x4_std, rtol=1e-5)

    def test_node_ordering_preservation(self, synthetic_gaussian_data):
        data = synthetic_gaussian_data["data"]
        # Intentionally order nodes in non-alphabetical sequence
        custom_model = LinearGaussianBayesianNetwork()
        custom_model.add_nodes_from(["x3", "x1", "x4", "x2"])
        custom_model.add_edges_from([("x1", "x2"), ("x2", "x3")])

        estimator = LinearGaussianMLE().fit(custom_model, data)
        learned_vars = [cpd.variable for cpd in estimator.parameters_]
        expected_vars = list(custom_model.nodes())
        assert learned_vars == expected_vars

    @pytest.mark.parametrize("chain_length", [2, 3, 5])
    def test_linear_chain_topologies(self, chain_length):
        rng = np.random.default_rng(123)
        n = 150
        data_dict = {"v0": rng.normal(1.0, 0.5, size=n)}
        edges = []
        for i in range(1, chain_length):
            prev_col = f"v{i - 1}"
            curr_col = f"v{i}"
            data_dict[curr_col] = 0.8 * data_dict[prev_col] + rng.normal(0.0, 0.3, size=n)
            edges.append((prev_col, curr_col))

        df = pd.DataFrame(data_dict)
        model = LinearGaussianBayesianNetwork(edges)
        estimator = LinearGaussianMLE(std_estimator="unbiased").fit(model, df)
        assert len(estimator.parameters_) == chain_length
        for i, cpd in enumerate(estimator.parameters_):
            if i == 0:
                assert cpd.evidence == []
                assert len(cpd.beta) == 1
            else:
                assert cpd.evidence == [f"v{i - 1}"]
                assert len(cpd.beta) == 2


class TestLinearGaussianMLEMissingDataAndEdgeCases:
    def test_dropna_with_missing_values_warning(self):
        rng = np.random.default_rng(999)
        n = 100
        x = rng.normal(0.0, 1.0, size=n)
        y = 1.5 * x + rng.normal(0.0, 0.5, size=n)
        df = pd.DataFrame({"x": x, "y": y})

        # Inject NaNs into 5 rows
        df.loc[10:14, "x"] = np.nan

        model = LinearGaussianBayesianNetwork([("x", "y")])
        estimator = LinearGaussianMLE()

        with pytest.warns(UserWarning, match="5 rows with missing values were dropped"):
            estimator.fit(model, df)

        # Verify that estimation completes correctly on the remaining 95 rows
        complete_df = df.dropna()
        cpd_y = next(cpd for cpd in estimator.parameters_ if cpd.variable == "y")
        expected_reg = LinearRegression().fit(complete_df[["x"]], complete_df["y"])
        np.testing.assert_allclose(cpd_y.beta[1:], expected_reg.coef_, rtol=1e-5)

    def test_deterministic_zero_residual_variance(self):
        # Edge case: perfect linear relationship with zero noise
        x = np.linspace(-5.0, 5.0, 50)
        y = 2.0 * x + 3.0
        df = pd.DataFrame({"x": x, "y": y})

        model = LinearGaussianBayesianNetwork([("x", "y")])
        estimator = LinearGaussianMLE(std_estimator="mle").fit(model, df)

        cpd_y = next(cpd for cpd in estimator.parameters_ if cpd.variable == "y")
        np.testing.assert_allclose(cpd_y.beta, [3.0, 2.0], atol=1e-6)
        assert cpd_y.std < 1e-6


class TestLinearGaussianMLEContractAndValidationFailures:
    def test_invalid_std_estimator_raises_value_error(self, synthetic_gaussian_data):
        data = synthetic_gaussian_data["data"]
        model = synthetic_gaussian_data["model"]

        with pytest.raises(ValueError, match="std_estimator must be one of {'mle', 'unbiased'}"):
            LinearGaussianMLE(std_estimator="invalid_mode").fit(model, data)

        with pytest.raises(ValueError, match="std_estimator must be one of {'mle', 'unbiased'}"):
            LinearGaussianMLE(std_estimator=None).fit(model, data)

    def test_missing_node_in_dataset_raises_value_error(self):
        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [2.0, 4.0, 6.0]})
        # Model specifies node x3 which is absent in df
        model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])

        estimator = LinearGaussianMLE()
        with pytest.raises(ValueError, match="Nodes detected in the model that are not present in the dataset"):
            estimator.fit(model, df)

    def test_unsupported_model_type_raises_not_implemented(self, synthetic_gaussian_data):
        data = synthetic_gaussian_data["data"]
        discrete_model = DiscreteBayesianNetwork([("x1", "x2")])

        estimator = LinearGaussianMLE()
        with pytest.raises(NotImplementedError, match="LinearGaussianMLE is only implemented"):
            estimator.fit(discrete_model, data)

    def test_sample_weight_rejection(self, synthetic_gaussian_data):
        data = synthetic_gaussian_data["data"]
        model = synthetic_gaussian_data["model"]
        weights = np.ones(len(data))

        estimator = LinearGaussianMLE()
        with pytest.raises(ValueError, match="LinearGaussianMLE does not support `sample_weight`"):
            estimator.fit(model, data, sample_weight=weights)

    def test_model_tag_contracts(self):
        estimator = LinearGaussianMLE()
        assert estimator.get_tag("supports_latent_variables") is False
        assert estimator.get_tag("supports_weighted_data") is False
        assert estimator.get_tag("supported_model_types") == (LinearGaussianBayesianNetwork,)
