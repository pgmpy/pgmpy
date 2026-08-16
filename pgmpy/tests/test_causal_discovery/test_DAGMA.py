"""
Tests for the sklearn-compatible DAGMALinear class in pgmpy.causal_discovery.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery.DAGMA import DAGMALinear

# Skip all tests if torch is not installed -- it is an optional dependency.
pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="torch is not installed. Install with: pip install pgmpy[torch]",
)


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
    X -> Y [beta=0.5]
    Y -> Z [beta=0.8]
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

    def test_compare_with_official_dagma_synthetic(self):
        """
        Compare pgmpy's DAGMALinear against official dagma on synthetic data generated identically to the official
        dagma test notebook (``dagma/examples/dagma_test.ipynb``).
        Uses Erdos-Renyi DAG with d=20 nodes, s0=20 edges, n=500 samples, Gaussian noise, and same random seed. Both
        implementations use least-squares loss with lambda1=0.02.
        """
        dagma_mod = pytest.importorskip("dagma.linear")
        dagma_utils = pytest.importorskip("dagma.utils")
        OfficialDagmaLinear = dagma_mod.DagmaLinear

        # 1. Generate synthetic data matching official test notebook
        dagma_utils.set_random_seed(1)
        n, d, s0 = 500, 20, 20
        graph_type, sem_type = "ER", "gauss"

        B_true = dagma_utils.simulate_dag(d, s0, graph_type)
        W_true = dagma_utils.simulate_parameter(B_true)
        X = dagma_utils.simulate_linear_sem(W_true, n, sem_type)

        # 2. Run official DAGMA
        model_official = OfficialDagmaLinear(loss_type="l2")
        W_est_official = model_official.fit(X, lambda1=0.02)

        # 3. Run pgmpy's DAGMA on same data (string column names required --
        #    _check_fit_data only sets feature_names_in_ for non-DataFrames;
        #    skbase's validate_data sets it only for string columns)
        col_names = [f"x{i}" for i in range(d)]
        X_df = pd.DataFrame(X, columns=col_names)
        est = DAGMALinear(lambda1=0.02, w_threshold=0.3)
        est.fit(X_df)

        # 4. Both should recover the true DAG edges
        true_edges = set(zip(*np.where(W_true != 0)))
        pgmpy_edges = {(col_names.index(u), col_names.index(v)) for u, v in est.causal_graph_.edges()}

        # True positive rate: pgmpy should recover most true edges
        tp = len(pgmpy_edges & true_edges)
        tpr = tp / len(true_edges) if true_edges else 0
        assert tpr >= 0.85, f"TPR={tpr:.2f} -- pgmpy recovered only {tp}/{len(true_edges)} true edges"

        # False discovery rate: pgmpy should not add many spurious edges
        fp = len(pgmpy_edges - true_edges)
        fdr = fp / len(pgmpy_edges) if pgmpy_edges else 0
        assert fdr <= 0.05, f"FDR={fdr:.2f} -- pgmpy added {fp} spurious edges"

        # 5. SHD between pgmpy and official should be small
        # (imported here: pgmpy.metrics pulls torch, which is an optional dependency)
        from pgmpy.metrics import SHD

        df_off = pd.DataFrame(W_est_official != 0, index=col_names, columns=col_names)
        nx_off = nx.from_pandas_adjacency(df_off, create_using=nx.DiGraph)
        dag_off = DAG(nx_off)

        shd_val = SHD()(true_causal_graph=dag_off, est_causal_graph=est.causal_graph_)
        assert shd_val == 0, f"SHD={shd_val} -- structures diverge more than expected"

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

    def test_domain_violation_recovery(self):
        """
        Test that optimization recovers gracefully from M-matrix domain violations.

        Uses a strict scalar s=0.5 (the default is the schedule
        ``[1.0, 0.9, 0.8, 0.7, 0.6]``) which tightens the M-matrix domain. The retry
        logic in ``_optimize()`` handles any violation by halving the learning rate and
        loosening s, producing a valid DAG instead of getting stuck.
        """
        dagma_utils = pytest.importorskip("dagma.utils")
        dagma_utils.set_random_seed(42)
        d, n = 10, 200
        B = dagma_utils.simulate_dag(d, 10, "ER")
        W = dagma_utils.simulate_parameter(B)
        X = dagma_utils.simulate_linear_sem(W, n, "gauss")

        est = DAGMALinear(s=0.5, lambda1=0.02, w_threshold=0.1)
        est.fit(pd.DataFrame(X, columns=[f"x{i}" for i in range(d)]))
        assert isinstance(est.causal_graph_, DAG)

    def test_analytical_gradient_matches_autograd(self):
        """Verify analytical gradient matches autograd to machine precision."""
        import torch

        d = 10
        torch.manual_seed(42)
        W = torch.randn(d, d, dtype=torch.float64) * 0.1
        cov = torch.eye(d, dtype=torch.float64)
        eye = torch.eye(d, dtype=torch.float64)
        mu, s, lambda1 = 1.0, 1.0, 0.05

        est = DAGMALinear(lambda1=lambda1, s=s)
        est.n_features_in_ = d

        # Analytical gradient
        grad_analytical, is_valid = est._gradient(W, mu, s, cov, eye)
        assert is_valid

        # Autograd gradient
        W_auto = W.clone().requires_grad_(True)
        loss = est._objective_value(W_auto, mu, s, cov, eye)
        loss.backward()
        grad_autograd = W_auto.grad

        # Should match to machine precision
        torch.testing.assert_close(grad_analytical, grad_autograd, atol=1e-12, rtol=1e-12)

    def test_warm_iter_parameter(self, continuous_data):
        """Test that warm_iter is respected and produces a valid DAG."""
        est = DAGMALinear(warm_iter=500, inner_iter=1000)
        est.fit(continuous_data)
        assert isinstance(est.causal_graph_, DAG)

    def test_s_schedule_list(self, continuous_data):
        """Test that s accepts a list schedule and produces a valid DAG."""
        est = DAGMALinear(s=[1.0, 0.95, 0.9, 0.85, 0.8])
        est.fit(continuous_data)
        assert isinstance(est.causal_graph_, DAG)

    def test_random_state_reproducibility(self, continuous_data):
        est1 = DAGMALinear(random_state=42)
        est1.fit(continuous_data)
        est2 = DAGMALinear(random_state=42)
        est2.fit(continuous_data)
        np.testing.assert_array_equal(est1.adjacency_matrix_, est2.adjacency_matrix_)

    def test_return_type_cpdag(self, continuous_data):
        est = DAGMALinear(return_type="cpdag")
        est.fit(continuous_data)
        assert isinstance(est.causal_graph_, PDAG)

    def test_score_with_shd(self, continuous_data):
        est = DAGMALinear()
        est.fit(continuous_data)
        # score() with a known true graph
        true_dag = DAG([("X", "Y"), ("Y", "Z")])
        score = est.score(X=None, true_graph=true_dag, metric="SHD")
        assert isinstance(score, (int, float))
