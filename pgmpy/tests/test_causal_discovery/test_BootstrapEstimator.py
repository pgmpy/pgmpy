import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import ADMG
from pgmpy.causal_discovery import ANM, PC, BootstrapEstimator, HillClimbSearch
from pgmpy.causal_discovery._base import BaseCausalDiscovery


@parametrize_with_checks(
    [BootstrapEstimator(estimator=HillClimbSearch(return_type="dag"), n_bootstraps=2, n_jobs=1)],
    expected_failed_checks=lambda estimator: {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    },
)
def test_bootstrap_estimator_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def rand_data():
    """Generates a 5-node dataset with the true causal graph A -> C <- B, C -> D, D -> E <- B."""
    np.random.seed(42)
    data = pd.DataFrame()
    data["A"] = np.random.randint(0, 2, size=500)
    data["B"] = np.random.randint(0, 2, size=500)
    data["C"] = (data["A"] & data["B"]) ^ np.random.binomial(1, 0.05, size=500)
    data["D"] = data["C"] ^ np.random.binomial(1, 0.05, size=500)
    data["E"] = (data["D"] & data["B"]) ^ np.random.binomial(1, 0.05, size=500)
    return data.astype("category")


def edge_type_counts(edges):
    """For "->", "<-" and "--", the number of bootstrap graphs with that edge for each pair, in the order of the index:
    AB, AC, AD, AE, BC, BD, BE, CD, CE, DE."""
    return [(edges == edge_type).sum(axis=1).tolist() for edge_type in ("->", "<-", "--")]


def adjacency_to_edges(adjacency):
    """The (row, column) pairs with a 1 in a binary adjacency matrix."""
    stacked = adjacency.stack()
    return set(stacked[stacked == 1].index)


def test_bootstrap_dag_estimator(rand_data):
    est = BootstrapEstimator(
        HillClimbSearch(return_type="dag"), n_bootstraps=10, warm_start=True, show_progress=False, seed=0
    )
    est.fit(rand_data)

    # `estimator_` is fitted on the full data and isn't one of the bootstrap graphs.
    full_fit = HillClimbSearch(return_type="dag").fit(rand_data)
    assert set(est.estimator_.causal_graph_.edges()) == set(full_fit.causal_graph_.edges())
    assert list(est.bootstrap_edges_.columns) == list(range(10))
    assert edge_type_counts(est.bootstrap_edges_) == [
        [3, 6, 0, 0, 5, 0, 8, 6, 2, 10],
        [4, 4, 0, 0, 5, 2, 2, 4, 0, 0],
        [0] * 10,
    ]

    # A - B is adjacent in 7 of the 10 graphs, mostly as B -> A. The tie between B -> C and C -> B goes to B -> C.
    expected_edges = {("A", "C"), ("B", "A"), ("B", "C"), ("B", "E"), ("C", "D"), ("D", "E")}
    assert set(est.causal_graph_.edges()) == expected_edges
    assert adjacency_to_edges(est.adjacency_matrix_) == expected_edges
    assert set(est.get_consensus_graph(threshold=0.3).edges()) == expected_edges

    # A - C, B - C and C - D are adjacent in every bootstrap graph, but in different orientations.
    expected_high_edges = {("A", "C"), ("B", "C"), ("B", "E"), ("C", "D"), ("D", "E")}
    assert set(est.get_consensus_graph(threshold=0.8).edges()) == expected_high_edges
    assert adjacency_to_edges(est.get_adjacency_matrix(threshold=0.8)) == expected_high_edges

    # Pairs that are never adjacent are left out even with threshold 0. B - D is only seen as D -> B, which would close
    # the cycle B -> C -> D -> B, so it is skipped.
    with pytest.warns(UserWarning, match=r"\('B', 'D'\)"):
        assert set(est.get_consensus_graph(threshold=0.0).edges()) == expected_edges | {("C", "E")}

    for invalid_threshold in [-0.5, 1.5]:
        with pytest.raises(ValueError, match="Threshold"):
            est.get_consensus_graph(threshold=invalid_threshold)
        with pytest.raises(ValueError, match="Threshold"):
            est.get_adjacency_matrix(threshold=invalid_threshold)

    # A warm start keeps the bootstrap samples, bootstrap edges and `estimator_`, and only fits the new samples.
    samples, edges, estimator = est.bootstrap_samples_.copy(), est.bootstrap_edges_.copy(), est.estimator_
    est.set_params(n_bootstraps=20).fit(rand_data)
    np.testing.assert_array_equal(est.bootstrap_samples_[:10], samples)
    pd.testing.assert_frame_equal(est.bootstrap_edges_[list(range(10))], edges)
    assert est.estimator_ is estimator
    assert edge_type_counts(est.bootstrap_edges_) == [
        [6, 13, 0, 0, 11, 1, 15, 15, 4, 18],
        [7, 7, 0, 0, 9, 4, 5, 5, 2, 2],
        [0] * 10,
    ]

    # Without new bootstrap samples, a warm start only forms the consensus graph again.
    est.set_params(threshold=0.8).fit(rand_data)
    assert est.bootstrap_edges_.shape == (10, 20)
    assert set(est.causal_graph_.edges()) == expected_high_edges

    # A warm start needs the same data and sample_fraction, and can't drop bootstrap samples.
    with pytest.raises(ValueError, match="different dataset size or sample_fraction"):
        est.fit(rand_data.iloc[:200])
    with pytest.raises(ValueError, match="different dataset features"):
        est.fit(rand_data.rename(columns={"A": "Z"}))
    with pytest.raises(ValueError, match="must be larger or equal to"):
        est.set_params(n_bootstraps=3).fit(rand_data)
    with pytest.raises(ValueError, match="different dataset size or sample_fraction"):
        est.set_params(sample_fraction=0.5).fit(rand_data)


def test_bootstrap_pdag_estimator(rand_data):
    est = BootstrapEstimator(PC(return_type="pdag"), n_bootstraps=10, show_progress=False, seed=42).fit(rand_data)

    assert edge_type_counts(est.bootstrap_edges_) == [
        [0, 9, 0, 0, 9, 0, 9, 9, 0, 9],
        [0] * 10,
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    ]
    expected_edges = {("A", "C"), ("B", "C"), ("B", "E"), ("C", "D"), ("D", "E")}
    assert set(est.causal_graph_.edges()) == expected_edges
    assert adjacency_to_edges(est.adjacency_matrix_) == expected_edges


@pytest.mark.parametrize(
    "params",
    [
        {"estimator": "pc"},
        {"threshold": -0.5},
        {"threshold": 1.5},
        {"sample_fraction": 0},
        {"sample_fraction": 1.5},
        {"n_bootstraps": 0},
        {"replace": False},
    ],
)
def test_bootstrap_fit_invalid_params(rand_data, params):
    est = BootstrapEstimator(**{"estimator": HillClimbSearch(return_type="dag"), "show_progress": False, **params})
    with pytest.raises(ValueError, match=list(params)[0]):
        est.fit(rand_data)


def test_bootstrap_sampling_and_wrapped_estimator(rand_data):
    # sklearn input tags come from the wrapped estimator, e.g. ANM doesn't accept categorical data.
    assert BootstrapEstimator(ANM()).__sklearn_tags__().input_tags.categorical is False
    assert BootstrapEstimator(PC()).__sklearn_tags__().input_tags.categorical is True

    est = BootstrapEstimator(
        HillClimbSearch(return_type="dag"),
        n_bootstraps=2,
        sample_fraction=0.5,
        replace=False,
        n_jobs=1,
        show_progress=False,
        seed=0,
    )
    est.fit(rand_data)
    assert est.bootstrap_samples_.shape == (2, 250)
    assert all(len(np.unique(sample)) == 250 for sample in est.bootstrap_samples_)

    # The wrapped estimator's own progress bars are turned off; the bootstrap shows a single one.
    assert est.estimator.show_progress is True
    assert est.estimator_.show_progress is False


def test_bootstrap_other_graph_types():
    class FixedADMG(BaseCausalDiscovery):
        def _fit(self, X):
            self.causal_graph_ = ADMG(edge_list=[("A", "B", "->"), ("A", "B", "<>"), ("C", "B", "->")])
            self.adjacency_matrix_ = self.causal_graph_.to_adjacency(nodelist=X.columns)
            return self

    data = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=["A", "B", "C"])
    est = BootstrapEstimator(FixedADMG(), n_bootstraps=3, n_jobs=1, show_progress=False).fit(data)

    # A directed and a bidirected edge between the same pair are recorded together.
    assert list(est.bootstrap_edges_.iloc[:, 0].cat.categories) == ["->,<>", "<-"]
    assert est.bootstrap_edges_[0].to_dict() == {("A", "B"): "->,<>", ("A", "C"): np.nan, ("B", "C"): "<-"}
    assert isinstance(est.causal_graph_, ADMG)
    assert sorted(est.causal_graph_.get_edges(data=True)) == [("A", "B", "->"), ("A", "B", "<>"), ("C", "B", "->")]
    assert est.adjacency_matrix_.loc["A", "B"] == ("->", "<>")

    # The weakest edge of a cycle is skipped with a warning.
    est.bootstrap_edges_ = pd.DataFrame(
        [["->", "->", "->", "->"], ["<-", "<-", None, None], ["->", "->", "->", None]],
        index=pd.MultiIndex.from_tuples([("A", "B"), ("A", "C"), ("B", "C")], names=["u", "v"]),
    ).astype(pd.CategoricalDtype(["->", "<-"]))
    with pytest.warns(UserWarning, match=r"\('A', 'C'\)"):
        graph = est.get_consensus_graph(threshold=0.5)
    assert sorted(graph.get_edges(data=True)) == [("A", "B", "->"), ("B", "C", "->")]
