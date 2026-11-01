"""Tests for the sklearn-compatible NOTEARS class in pgmpy.causal_discovery."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery import NOTEARS


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [NOTEARS(show_progress=False)],
    expected_failed_checks=expected_failed_checks,
)
def test_notears_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def chain_data():
    # Same draw used to freeze the xunzheng/notears linear reference weights below.
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(scale=0.3, size=n)
    z = -1.0 * y + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


class TestNOTEARS:
    def test_recovers_chain_and_matches_reference_weights(self, chain_data):
        original = chain_data.copy()
        est = NOTEARS(lambda1=0.1, loss_type="l2", w_threshold=0.3, show_progress=False)
        assert est.fit(chain_data) is est
        assert isinstance(est.causal_graph_, DAG)
        assert est.n_features_in_ == 3
        assert list(est.feature_names_in_) == ["X", "Y", "Z"]
        assert set(est.causal_graph_.edges()) == {("X", "Y"), ("Y", "Z")}
        pd.testing.assert_frame_equal(
            est.adjacency_matrix_,
            pd.DataFrame(
                [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
                index=chain_data.columns,
                columns=chain_data.columns,
            ),
        )
        # Frozen output of xunzheng/notears linear.notears_linear (Apache-2.0) on this fixture,
        # lambda1=0.1, loss_type='l2', max_iter=100, w_threshold=0.3. Mean-center only; no std scaling.
        reference_W = np.array(
            [
                [0.0, 1.3458112924116241, 0.0],
                [0.0, 0.0, -0.9525942598216143],
                [0.0, 0.0, 0.0],
            ]
        )
        estimated_W = nx.to_pandas_adjacency(est.causal_graph_, nodelist=["X", "Y", "Z"], weight="weight").to_numpy()
        np.testing.assert_allclose(estimated_W, reference_W, atol=1e-5, rtol=1e-5)
        pd.testing.assert_frame_equal(chain_data, original)

        # Official W at w_threshold=0 keeps tiny reverse edges and is not a DAG.
        # Remaining cyclic edges are inserted by decreasing |weight|.
        est0 = NOTEARS(lambda1=0.1, loss_type="l2", w_threshold=0.0, show_progress=False)
        assert set(est0.fit(chain_data).causal_graph_.edges()) == {("X", "Y"), ("Y", "Z")}
        estimated_W0 = nx.to_pandas_adjacency(est0.causal_graph_, nodelist=["X", "Y", "Z"], weight="weight").to_numpy()
        np.testing.assert_allclose(estimated_W0, reference_W, atol=1e-5, rtol=1e-5)

    def test_invalid_inputs_raise(self):
        data = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        with pytest.raises(ValueError, match="loss_type must be one of"):
            NOTEARS(loss_type="invalid", show_progress=False).fit(data)
        with pytest.raises(ValueError, match="lambda1 must be non-negative"):
            NOTEARS(lambda1=-0.1, show_progress=False).fit(data)
        with pytest.raises(ValueError, match="max_iter must be a positive integer"):
            NOTEARS(max_iter=0, show_progress=False).fit(data)
        with pytest.raises(ValueError, match="w_threshold must be non-negative"):
            NOTEARS(w_threshold=-0.1, show_progress=False).fit(data)
        with pytest.raises(ValueError, match="binary"):
            NOTEARS(loss_type="logistic", show_progress=False).fit(pd.DataFrame({"A": [0, 1, 2], "B": [1, 0, 1]}))
        with pytest.raises(ValueError, match="non-negative"):
            NOTEARS(loss_type="poisson", show_progress=False).fit(pd.DataFrame({"A": [0, 1, -1], "B": [1, 0, 1]}))

    def test_logistic_and_poisson_match_reference_weights(self):
        rng = np.random.default_rng(0)
        n = 80
        x = rng.integers(0, 2, size=n)
        y = np.where(x == 1, rng.binomial(1, 0.9, size=n), rng.binomial(1, 0.1, size=n))
        logistic = NOTEARS(loss_type="logistic", show_progress=False).fit(pd.DataFrame({"X": x, "Y": y}))
        assert isinstance(logistic.causal_graph_, DAG)
        np.testing.assert_allclose(
            nx.to_pandas_adjacency(logistic.causal_graph_, nodelist=["X", "Y"], weight="weight").to_numpy(),
            np.array([[0.0, 0.0], [1.0348640584572018, 0.0]]),
            atol=1e-5,
            rtol=1e-5,
        )

        counts = pd.DataFrame({"A": rng.poisson(2, size=n), "B": rng.poisson(3, size=n)})
        poisson = NOTEARS(loss_type="poisson", show_progress=False).fit(counts)
        assert isinstance(poisson.causal_graph_, DAG)
        assert set(poisson.causal_graph_.edges()) == set()
        np.testing.assert_array_equal(poisson.adjacency_matrix_.to_numpy(), np.zeros((2, 2)))

    def test_expert_knowledge_not_supported(self):
        with pytest.raises(TypeError, match="expert_knowledge"):
            NOTEARS(expert_knowledge=None)
