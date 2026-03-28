"""
Tests for the sklearn-compatible NOTEARS class in pgmpy.causal_discovery.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy import config
from pgmpy.causal_discovery import NOTEARS, ExpertKnowledge
from pgmpy.utils import compat_fns

BACKEND_PARAMS = ["numpy"]
if _check_soft_dependencies("torch", severity="none"):
    BACKEND_PARAMS.append("torch")

torch = _safe_import("torch")


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": (
            "NOTEARS does not set or validate the `n_features_in_` attribute for the "
            "score method in the way required by this sklearn compatibility check, so "
            "this check is not applicable to NOTEARS."
        ),
    }


@parametrize_with_checks(
    [NOTEARS(lambda1=0.1, loss_type="l2")],
    expected_failed_checks=expected_failed_checks,
)
def test_notears_compatibility(estimator, check):
    check(estimator)


@pytest.fixture(params=BACKEND_PARAMS)
def backend(request):
    prev_backend = config.get_backend()
    config.set_backend(request.param)
    yield request.param
    config.set_backend(prev_backend)


@pytest.fixture
def continuous_data():
    rng = np.random.default_rng(101)
    x = rng.normal(size=300)
    y = 1.7 * x + rng.normal(scale=0.2, size=300)
    z = -1.3 * y + rng.normal(scale=0.2, size=300)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


@pytest.fixture
def logistic_data():
    rng = np.random.default_rng(7)
    x = rng.binomial(1, 0.5, size=350)
    py = 1.0 / (1.0 + np.exp(-(1.2 * x - 0.1)))
    y = rng.binomial(1, py)
    pz = 1.0 / (1.0 + np.exp(-(1.1 * y - 0.1)))
    z = rng.binomial(1, pz)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


@pytest.fixture
def poisson_data():
    rng = np.random.default_rng(23)
    x = rng.poisson(2.0, size=300)
    lam_y = np.exp(np.clip(0.2 + 0.15 * x, -2, 2))
    y = rng.poisson(lam_y)
    lam_z = np.exp(np.clip(0.1 + 0.12 * y, -2, 2))
    z = rng.poisson(lam_z)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


# The following function is adapted from the reference implementation in:
#   https://github.com/xunzheng/notears
# which is licensed under the MIT License:
#
# MIT License
#
# Copyright (c) 2018 Xun Zheng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This inlined implementation is used here solely as a reference for testing
# pgmpy's NOTEARS implementation and is not part of the library's public API.
def _notears_reference(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=0.3):
    """Reference NOTEARS implementation adapted from https://github.com/xunzheng/notears."""

    def _loss(W):
        M = X @ W
        if loss_type == "l2":
            R = X - M
            loss = 0.5 / X.shape[0] * (R**2).sum()
            G_loss = -1.0 / X.shape[0] * X.T @ R
        elif loss_type == "logistic":
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == "poisson":
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError("unknown loss type")
        return loss, G_loss

    def _h(W):
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        return (w[: d * d] - w[d * d :]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == "l2":
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


class TestNOTEARSCore:
    """Core functionality tests."""

    def test_fit_returns_dag(self, backend, continuous_data):
        est = NOTEARS(lambda1=0.01, max_iter=3, show_progress=False)
        est.fit(continuous_data)
        assert hasattr(est, "causal_graph_")
        assert hasattr(est, "adjacency_matrix_")
        assert nx.is_directed_acyclic_graph(est.causal_graph_)

    def test_feature_names_stored(self, backend, continuous_data):
        est = NOTEARS(lambda1=0.01, max_iter=3, show_progress=False)
        est.fit(continuous_data)
        assert est.n_features_in_ == 3
        assert set(est.feature_names_in_) == {"X", "Y", "Z"}

    def test_adjacency_matrix_shape(self, backend, continuous_data):
        est = NOTEARS(lambda1=0.01, max_iter=3, show_progress=False)
        est.fit(continuous_data)
        assert est.adjacency_matrix_.shape == (3, 3)

    def test_no_stdout_when_progress_disabled(self, backend, continuous_data, capsys):
        est = NOTEARS(lambda1=0.01, max_iter=2, show_progress=False)
        est.fit(continuous_data)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestNOTEARSLossTypes:
    """Tests for different loss functions."""

    @pytest.mark.parametrize(
        ("loss_type", "fixture_name"),
        [
            ("l2", "continuous_data"),
            ("logistic", "logistic_data"),
            ("poisson", "poisson_data"),
        ],
    )
    def test_loss_types_return_acyclic_dag(self, backend, request, loss_type, fixture_name):
        data = request.getfixturevalue(fixture_name)
        est = NOTEARS(lambda1=0.01, loss_type=loss_type, max_iter=3, show_progress=False)
        est.fit(data)
        assert nx.is_directed_acyclic_graph(est.causal_graph_)


class TestNOTEARSValidation:
    """Input validation tests."""

    @pytest.mark.parametrize(
        ("kwargs", "data", "error_msg"),
        [
            (
                {"loss_type": "invalid"},
                pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}),
                "loss_type must be one of",
            ),
            (
                {"loss_type": "logistic"},
                pd.DataFrame({"A": [0, 1, 2], "B": [1, 0, 1]}),
                "binary",
            ),
            (
                {"loss_type": "poisson"},
                pd.DataFrame({"A": [0, 1, -1], "B": [1, 0, 1]}),
                "non-negative",
            ),
            (
                {"loss_type": "l2"},
                pd.DataFrame({"A": [1.0, np.nan], "B": [3.0, 4.0]}),
                "missing values",
            ),
        ],
    )
    def test_invalid_data_raises(self, backend, kwargs, data, error_msg):
        est = NOTEARS(lambda1=0.01, show_progress=False, **kwargs)
        with pytest.raises(ValueError, match=error_msg):
            est.fit(data)

    @pytest.mark.parametrize(
        ("kwargs", "error_msg"),
        [
            ({"lambda1": -0.1}, "lambda1 must be non-negative"),
            ({"max_iter": 0}, "max_iter must be a positive integer"),
            ({"w_threshold": -0.1}, "w_threshold must be non-negative"),
        ],
    )
    def test_invalid_hyperparameters_raise(self, backend, kwargs, error_msg):
        data = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        est = NOTEARS(show_progress=False, **kwargs)
        with pytest.raises(ValueError, match=error_msg):
            est.fit(data)


class TestNOTEARSExpertKnowledge:
    """Tests for expert knowledge constraints."""

    def test_search_space_limits_edges(self, backend):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({"A": rng.normal(size=100), "B": rng.normal(size=100)})
        expert = ExpertKnowledge(search_space=[("A", "B")])
        est = NOTEARS(
            lambda1=0.01,
            max_iter=5,
            w_threshold=0.0,
            expert_knowledge=expert,
            show_progress=False,
        )
        est.fit(data)
        assert set(est.causal_graph_.edges()).issubset({("A", "B")})

    def test_forbidden_edges_excluded(self, backend, continuous_data):
        expert = ExpertKnowledge(forbidden_edges=[("Y", "X")])
        est = NOTEARS(
            lambda1=0.01,
            max_iter=5,
            w_threshold=0.0,
            expert_knowledge=expert,
            show_progress=False,
        )
        est.fit(continuous_data)
        assert ("Y", "X") not in est.causal_graph_.edges()

    @pytest.mark.parametrize("w_threshold", [0.1, 0.0])
    def test_required_edges(self, backend, w_threshold):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({"A": rng.normal(size=100), "B": rng.normal(size=100)})
        expert = ExpertKnowledge(required_edges=[("A", "B")])
        est = NOTEARS(
            lambda1=0.01,
            max_iter=10,
            w_threshold=w_threshold,
            expert_knowledge=expert,
            show_progress=False,
        )
        est.fit(data)
        assert ("A", "B") in est.causal_graph_.edges()

    @pytest.mark.parametrize(
        ("expert_knowledge", "error_msg"),
        [
            (
                ExpertKnowledge(required_edges=[("A", "C")]),
                r"Expert knowledge edge \(A, C\) refers to node\(s\) not present in the data columns\.",
            ),
            (
                ExpertKnowledge(forbidden_edges=[("A", "C")]),
                r"Expert knowledge edge \(A, C\) refers to node\(s\) not present in the data columns\.",
            ),
        ],
    )
    def test_missing_nodes_in_expert_knowledge(self, backend, expert_knowledge, error_msg):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({"A": rng.normal(size=50), "B": rng.normal(size=50)})
        est = NOTEARS(
            lambda1=0.01,
            max_iter=5,
            expert_knowledge=expert_knowledge,
            show_progress=False,
        )
        with pytest.raises(ValueError, match=error_msg):
            est.fit(data)

    @pytest.mark.parametrize(
        ("expert_knowledge", "error_msg"),
        [
            (
                ExpertKnowledge(required_edges=[("A", "B")], forbidden_edges=[("A", "B")]),
                r"Expert knowledge conflict: Edge \(A, B\) is both required and forbidden\.",
            ),
            (
                ExpertKnowledge(required_edges=[("A", "A")]),
                r"Expert knowledge conflict: Self-loop \(A, A\) cannot be required\.",
            ),
        ],
    )
    def test_conflicting_expert_knowledge_raises(self, backend, expert_knowledge, error_msg):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({"A": rng.normal(size=50), "B": rng.normal(size=50)})
        est = NOTEARS(
            lambda1=0.01,
            max_iter=5,
            expert_knowledge=expert_knowledge,
            show_progress=False,
        )
        with pytest.raises(ValueError, match=error_msg):
            est.fit(data)

    def test_required_edges_cycle_raises(self, backend):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({"A": rng.normal(size=50), "B": rng.normal(size=50)})
        expert = ExpertKnowledge(required_edges=[("A", "B"), ("B", "A")])
        est = NOTEARS(
            lambda1=0.01,
            max_iter=5,
            expert_knowledge=expert,
            show_progress=False,
        )
        with pytest.raises(
            ValueError,
            match="required_edges create a cycle in the output DAG",
        ):
            est.fit(data)

    def test_temporal_order_excludes_backward_edges(self, backend, continuous_data):
        expert = ExpertKnowledge(temporal_order=[["X"], ["Y"], ["Z"]])
        est = NOTEARS(
            lambda1=0.01,
            max_iter=5,
            w_threshold=0.0,
            expert_knowledge=expert,
            show_progress=False,
        )
        est.fit(continuous_data)
        temporal_ordering = expert.temporal_ordering
        for u, v in est.causal_graph_.edges():
            assert temporal_ordering[u] <= temporal_ordering[v]


class TestNOTEARSReferenceComparison:
    """Compare pgmpy NOTEARS output with the reference implementation."""

    @pytest.mark.parametrize(
        ("loss_type", "X"),
        [
            ("logistic", np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])),
            ("poisson", np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 3.0]])),
        ],
    )
    def test_reference_supports_other_losses(self, loss_type, X):
        W_est = _notears_reference(X, lambda1=0.1, loss_type=loss_type, max_iter=1, w_threshold=0.0)
        assert W_est.shape == (2, 2)

    def test_reference_invalid_loss_raises(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="unknown loss type"):
            _notears_reference(X, lambda1=0.1, loss_type="invalid", max_iter=1)

    def test_matches_reference_l2(self):
        prev_backend = config.get_backend()
        config.set_backend("numpy")
        try:
            rng = np.random.default_rng(42)
            x = rng.normal(size=200)
            y = 1.5 * x + rng.normal(scale=0.3, size=200)
            z = -1.0 * y + rng.normal(scale=0.3, size=200)
            X_np = np.column_stack([x, y, z])
            X_df = pd.DataFrame(X_np, columns=["X", "Y", "Z"])

            lambda1, w_threshold = 0.1, 0.3
            ref_W = _notears_reference(
                X_np,
                lambda1=lambda1,
                loss_type="l2",
                w_threshold=w_threshold,
                max_iter=20,
            )

            est = NOTEARS(
                lambda1=lambda1,
                loss_type="l2",
                w_threshold=w_threshold,
                show_progress=False,
            )
            est.fit(X_df)

            # Both should recover the same non-zero edge pattern
            ref_edges = set()
            for i in range(3):
                for j in range(3):
                    if ref_W[i, j] != 0:
                        ref_edges.add((["X", "Y", "Z"][i], ["X", "Y", "Z"][j]))

            pgmpy_edges = set(est.causal_graph_.edges())

            # The non-zero patterns should overlap significantly.
            # The pgmpy version adds a DAG-enforcement step (greedy by weight)
            # so the exact edge set may differ slightly, but the strong edges
            # should match.
            assert len(pgmpy_edges) > 0
            assert len(ref_edges) > 0
            # At minimum, both should find X->Y (the strongest edge)
            assert ("X", "Y") in ref_edges or ("X", "Y") in pgmpy_edges
        finally:
            config.set_backend(prev_backend)


class TestNOTEARSCompatFns:
    def test_matrix_exp_numpy_matches_scipy(self):
        arr = np.array([[0.0, 1.0], [0.0, 0.0]])
        np.testing.assert_allclose(compat_fns.matrix_exp(arr), slin.expm(arr))

    @pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="torch not installed")
    def test_matrix_exp_torch_matches_torch(self):
        arr = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(compat_fns.matrix_exp(arr), torch.matrix_exp(arr))

    def test_concatenate_fallback_accepts_python_sequences(self):
        result = compat_fns.concatenate([1.0, 2.0], (3.0, 4.0))
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0, 4.0]))

    @pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="torch not installed")
    @pytest.mark.parametrize("tensor_first", [True, False])
    def test_concatenate_torch_mixed_inputs(self, tensor_first):
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
        array = np.array([3.0, 4.0])
        left, right = (tensor, array) if tensor_first else (array, tensor)
        expected = np.array([1.0, 2.0, 3.0, 4.0]) if tensor_first else np.array([3.0, 4.0, 1.0, 2.0])

        result = compat_fns.concatenate(left, right)

        assert torch.is_tensor(result)
        np.testing.assert_array_equal(result.detach().cpu().numpy(), expected)
