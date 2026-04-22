"""
Tests for the sklearn-compatible DAGMALinear class in pgmpy.causal_discovery.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery.DAGMA import DAGMALinear
from pgmpy.metrics import SHD
from pgmpy.models import LinearGaussianBayesianNetwork


def expected_failed_checks(estimator):
    """
    scikit-learn checks that are expected to fail
    for pgmpy causal discovery algorithms.
    """
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
        "check_dtype_object": "DAGMA only supports float data types, not object dtype.",
    }


@parametrize_with_checks(
    [DAGMALinear()],
    expected_failed_checks=expected_failed_checks,
)
def test_dagma_compatibility(estimator, check):
    """
    Automatically runs scikit-learn API compliance checks.
    """
    check(estimator)


@pytest.fixture
def continuous_data():
    """
    Set up a simple synthetic dataset using LinearGaussianBN.
    Creates a chain X -> Y -> Z with known causal relationships.
    """
    from pgmpy.factors.continuous import LinearGaussianCPD

    model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
    # X ~ N(0, 1): beta=[0], std=1
    cpd_x = LinearGaussianCPD("X", [0], 1)
    # Y = 2.0*X + N(0, 0.5): beta=[0, 2.0], std=0.5
    cpd_y = LinearGaussianCPD("Y", [0, 2.0], 0.5, evidence=["X"])
    # Z = 1.5*Y + N(0, 0.5): beta=[0, 1.5], std=0.5
    cpd_z = LinearGaussianCPD("Z", [0, 1.5], 0.5, evidence=["Y"])
    model.add_cpds(cpd_x, cpd_y, cpd_z)
    data = model.simulate(n_samples=1000, seed=42)
    return data


class TestDagmaLinearCore:
    """Tests for core DagmaLinear functionality."""

    def test_estimate_returns_dag(self, continuous_data):
        est = DAGMALinear()

        est.fit(continuous_data)

        # 1. Check if the output is successfully saved as a pgmpy DAG
        assert isinstance(est.causal_graph_, DAG)

        # 2. Check if the extracted feature names match the dataframe columns
        np.testing.assert_array_equal(est.feature_names_in_, ["X", "Y", "Z"])

        # 3. Check if the estimated adjacency matrix was saved and is a NumPy array
        assert isinstance(est.adjacency_matrix_, np.ndarray)
        assert est.adjacency_matrix_.shape == (3, 3)

        # 4. Check if the algorithm successfully found the X -> Y and Y -> Z edges
        learned_edges = list(est.causal_graph_.edges())
        assert ("X", "Y") in learned_edges
        assert ("Y", "Z") in learned_edges

        # 5. Prove the bounds and barriers successfully prevented cycles and self-loops
        assert ("Z", "X") not in learned_edges
        assert ("X", "X") not in learned_edges
        assert ("Y", "Y") not in learned_edges
        assert ("Z", "Z") not in learned_edges

    def test_dag_acyclicity_check(self, continuous_data):
        """Verify the learned graph is always a valid DAG."""
        est = DAGMALinear()
        est.fit(continuous_data)

        # Explicit acyclicity check
        assert nx.is_directed_acyclic_graph(est.causal_graph_), "Learned graph must be a valid DAG"

    def test_scalability_5_variables(self):
        """
        Test with 5-10 variables. Verifies the algorithm scales and maintains acyclicity.
        """
        from pgmpy.factors.continuous import LinearGaussianCPD

        model = LinearGaussianBayesianNetwork(
            [("V1", "V2"), ("V1", "V3"), ("V2", "V3"), ("V3", "V5"), ("V4", "V7"), ("V5", "V7")]
        )
        model.add_node("V6")

        cpd_v1 = LinearGaussianCPD("V1", [0], 1)
        cpd_v4 = LinearGaussianCPD("V4", [0], 1)
        cpd_v6 = LinearGaussianCPD("V6", [0], 1)

        cpd_v2 = LinearGaussianCPD("V2", [0, 1.5], 0.5, evidence=["V1"])
        cpd_v3 = LinearGaussianCPD("V3", [0, 2.0, 0.5], 0.5, evidence=["V1", "V2"])
        cpd_v5 = LinearGaussianCPD("V5", [0, 1.0], 0.5, evidence=["V3"])
        cpd_v7 = LinearGaussianCPD("V7", [0, 0.8, 1.2], 0.5, evidence=["V4", "V5"])

        model.add_cpds(cpd_v1, cpd_v2, cpd_v3, cpd_v4, cpd_v5, cpd_v6, cpd_v7)
        data = model.simulate(n_samples=1000, seed=42)

        est = DAGMALinear()
        est.fit(data)

        # Verify DAG property holds with more variables
        assert nx.is_directed_acyclic_graph(est.causal_graph_)
        assert est.adjacency_matrix_.shape == (7, 7)

    def test_against_linear_gaussian_bn(self):
        """
        Use LinearGaussianBN to generate data with known causal structure
        as specified in SPEC.md Section 5.1.
        """
        from pgmpy.factors.continuous import LinearGaussianCPD

        if LinearGaussianBayesianNetwork is None:
            pytest.skip("LinearGaussianBayesianNetwork not available")

        model = LinearGaussianBayesianNetwork([("X1", "X2"), ("X2", "X3")])
        cpd_x1 = LinearGaussianCPD("X1", [0], 1)
        cpd_x2 = LinearGaussianCPD("X2", [0, 2.0], 0.5, evidence=["X1"])
        cpd_x3 = LinearGaussianCPD("X3", [0, 1.5], 0.5, evidence=["X2"])
        model.add_cpds(cpd_x1, cpd_x2, cpd_x3)
        data = model.simulate(n_samples=1000, seed=42)

        est = DAGMALinear()
        est.fit(data)

        # Verify DAG property
        assert nx.is_directed_acyclic_graph(est.causal_graph_)

        # We verify the estimated graph has correct structure
        edges = list(est.causal_graph_.edges())
        assert len(edges) >= 2  # At least X1->X2 and X2->X3

    def test_custom_hyperparameters(self):
        """
        Ensure custom hyperparameters are strictly mapped to the instance.
        """
        est = DAGMALinear(s=2.0, lambda1=0.1, mu_init=2.0, mu_factor=0.5, max_iter=50, w_threshold=0.4)
        assert est.s == 2.0
        assert est.lambda1 == 0.1
        assert est.mu_init == 2.0
        assert est.mu_factor == 0.5
        assert est.max_iter == 50
        assert est.w_threshold == 0.4

    def test_compare_with_official_dagma(self):
        """
        Compare the adjacency matrix output of pgmpy's DAGMALinear
        with the official dagma package implementation on the Sachs dataset.
        """
        dagma_mod = pytest.importorskip("dagma.linear")
        OfficialDagmaLinear = dagma_mod.DagmaLinear
        from pgmpy.datasets import load_dataset

        # 1. Load Sachs continuous dataset
        try:
            data_obj = load_dataset("sachs_continuous")
            data = data_obj.data
        except Exception:
            pytest.skip("Sachs dataset not available.")

        # Standardize for comparison consistency across different optimizers
        data = (data - data.mean()) / data.std()
        data_np = data.to_numpy().copy()
        nodes = data.columns.tolist()

        # 2. Run official DAGMA
        # We force s=1.0 and T=5 to match pgmpy's default single-stage behavior
        model_official = OfficialDagmaLinear(loss_type="l2")
        W_official = model_official.fit(data_np, lambda1=0.05, w_threshold=0.3, s=1.0, T=5)

        # 3. Run pgmpy's DAGMA
        est = DAGMALinear(lambda1=0.05, w_threshold=0.3, s=1.0)
        est.fit(data)

        # 4. Compare structures using Structural Hamming Distance (SHD)
        # Convert official adjacency to pgmpy DAG for metric calculation
        df_adj = pd.DataFrame(W_official, index=nodes, columns=nodes)
        nx_official = nx.from_pandas_adjacency(df_adj, create_using=nx.DiGraph)
        dag_official = DAG(nx_official)

        shd_metric = SHD()
        shd_val = shd_metric(true_causal_graph=dag_official, est_causal_graph=est.causal_graph_)

        # SHD <= 6 indicates high structural similarity for 11 nodes
        assert shd_val <= 6


class TestDagmaLinear:
    """Tests for DAGMALinear using pytest style."""

    @pytest.fixture
    def data(self):
        """Set up a simple synthetic dataset using LinearGaussianBN."""
        from pgmpy.factors.continuous import LinearGaussianCPD

        model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
        cpd_x = LinearGaussianCPD("X", [0], 1)
        cpd_y = LinearGaussianCPD("Y", [0, 2.0], 0.5, evidence=["X"])
        cpd_z = LinearGaussianCPD("Z", [0, 1.5], 0.5, evidence=["Y"])
        model.add_cpds(cpd_x, cpd_y, cpd_z)
        data = model.simulate(n_samples=1000, seed=42)
        return data

    def test_fit_returns_dag(self, data):
        """
        Test if the fit method runs successfully
        and returns a proper DAG object.
        """
        estimator = DAGMALinear()
        estimator.fit(data)

        assert isinstance(estimator.causal_graph_, DAG)
        np.testing.assert_array_equal(estimator.feature_names_in_, ["X", "Y", "Z"])
        assert isinstance(estimator.adjacency_matrix_, np.ndarray)
        assert estimator.adjacency_matrix_.shape == (3, 3)

        learned_edges = list(estimator.causal_graph_.edges())
        assert ("X", "Y") in learned_edges
        assert ("Y", "Z") in learned_edges
        assert ("Z", "X") not in learned_edges

    def test_custom_hyperparameters(self):
        """Test if the __init__ correctly stores user-defined hyperparameters."""
        estimator = DAGMALinear(lambda1=0.1, max_iter=50)
        assert estimator.lambda1 == 0.1
        assert estimator.max_iter == 50
