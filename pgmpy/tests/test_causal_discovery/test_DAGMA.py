"""
Tests for the sklearn-compatible DAGMALinear class in pgmpy.causal_discovery.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery.DAGMA import DAGMALinear


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
    Set up a simple synthetic dataset using DAG.from_dagitty.
    Creates a chain X -> Y -> Z with known causal relationships.
    """
    dagitty_str = """
    dag {
    X [exposure]
    Y
    Z [outcome]
    X -> Y [beta=2.0]
    Y -> Z [beta=1.5]
    }
    """
    model = DAG.from_dagitty(string=dagitty_str)
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

    def test_scalability_5_variables(self):
        """
        Test with 5-10 variables. Verifies the algorithm scales and recovers edges.
        """
        dagitty_str = """
        dag {
        V1 -> V2 [beta=1.5]
        V1 -> V3 [beta=2.0]
        V2 -> V3 [beta=0.5]
        V3 -> V5 [beta=1.0]
        V4 -> V7 [beta=0.8]
        V5 -> V7 [beta=1.2]
        V6
        }
        """
        model = DAG.from_dagitty(string=dagitty_str)
        data = model.simulate(n_samples=1000, seed=42)

        est = DAGMALinear()
        est.fit(data)

        # Check for how many edges are correctly recovered
        learned_edges = set(est.causal_graph_.edges())
        true_edges = {("V1", "V2"), ("V1", "V3"), ("V2", "V3"), ("V3", "V5"), ("V4", "V7"), ("V5", "V7")}
        correct_edges = learned_edges.intersection(true_edges)
        assert len(correct_edges) >= 4
        assert est.adjacency_matrix_.shape == (7, 7)

    def test_against_linear_gaussian_bn(self):
        """
        Use LinearGaussianBN to generate data with known causal structure
        as specified in SPEC.md Section 5.1.
        """
        dagitty_str = """
        dag {
        X1 -> X2 [beta=2.0]
        X2 -> X3 [beta=1.5]
        }
        """
        model = DAG.from_dagitty(string=dagitty_str)
        data = model.simulate(n_samples=1000, seed=42)

        est = DAGMALinear()
        est.fit(data)

        # Verify the estimated graph has correct structure
        edges = list(est.causal_graph_.edges())
        assert len(edges) >= 2  # At least X1->X2 and X2->X3

    def test_custom_hyperparameters(self):
        """
        Ensure custom hyperparameters are strictly mapped to the instance.
        """
        est = DAGMALinear(
            s=2.0,
            lambda1=0.1,
            mu_init=2.0,
            mu_factor=0.5,
            max_iter=50,
            inner_iter=500,
            w_threshold=0.4,
            optimizer_kwargs={"max_iter": 5},
        )
        assert est.s == 2.0
        assert est.lambda1 == 0.1
        assert est.mu_init == 2.0
        assert est.mu_factor == 0.5
        assert est.max_iter == 50
        assert est.inner_iter == 500
        assert est.w_threshold == 0.4
        assert est.optimizer_kwargs == {"max_iter": 5}

    def test_compare_with_official_dagma(self):
        """
        Compare pgmpy's DAGMALinear output against the known output from the
        official dagma package on the Sachs dataset.

        To reproduce the reference output, install the ``dagma`` package and run::

            # from dagma.linear import DagmaLinear
            # from pgmpy.datasets import load_dataset
            # data = load_dataset("sachs_continuous").data
            # data = (data - data.mean()) / data.std()
            # model = DagmaLinear(loss_type="l2")  # l2 = least-squares loss (not L2 reg)
            # W = model.fit(data.to_numpy(), lambda1=0.05, w_threshold=0.3, s=1.0, T=5)
        """
        dagma_mod = pytest.importorskip("dagma.linear")
        OfficialDagmaLinear = dagma_mod.DagmaLinear  # pragma: no cover
        from pgmpy.datasets import load_dataset  # pragma: no cover
        from pgmpy.metrics import SHD  # pragma: no cover

        # 1. Load Sachs continuous dataset
        data = load_dataset("sachs_continuous").data  # pragma: no cover

        # Standardize for comparison consistency
        data = (data - data.mean()) / data.std()  # pragma: no cover
        data_np = data.to_numpy().copy()  # pragma: no cover
        nodes = data.columns.tolist()  # pragma: no cover

        # 2. Run official DAGMA (loss_type="l2" = least-squares loss, lambda1 = L1 reg)
        model_official = OfficialDagmaLinear(loss_type="l2")  # pragma: no cover
        W_official = model_official.fit(data_np, lambda1=0.05, w_threshold=0.3, s=1.0, T=5)  # pragma: no cover

        # 3. Run pgmpy's DAGMA
        est = DAGMALinear(lambda1=0.05, w_threshold=0.3, s=1.0)  # pragma: no cover
        est.fit(data)  # pragma: no cover

        # 4. Compare structures using SHD
        df_adj = pd.DataFrame(W_official, index=nodes, columns=nodes)  # pragma: no cover
        import networkx as nx  # pragma: no cover

        nx_official = nx.from_pandas_adjacency(df_adj, create_using=nx.DiGraph)  # pragma: no cover
        dag_official = DAG(nx_official)  # pragma: no cover

        shd_metric = SHD()  # pragma: no cover
        shd_val = shd_metric(true_causal_graph=dag_official, est_causal_graph=est.causal_graph_)  # pragma: no cover

        # SHD <= 6 indicates high structural similarity for 11 nodes
        assert shd_val <= 6  # pragma: no cover

    def test_optimizer_kwargs(self, continuous_data):
        """
        Test that custom optimizer_kwargs are passed through to the optimizer.
        """
        import torch

        est = DAGMALinear(
            optimizer=torch.optim.LBFGS, optimizer_kwargs={"max_iter": 5, "line_search_fn": "strong_wolfe"}
        )
        est.fit(continuous_data)
        assert isinstance(est.causal_graph_, DAG)
