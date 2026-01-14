import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery.TOPIC import TOPIC

""" Utils """


def fake_score_fn_factory():
    base = {"A": 0.0, "B": 10.0, "C": 20.0}
    w = {("A", "B"): 1.0, ("C", "B"): -2.0, ("B", "C"): 3.0, ("A", "C"): 4.0}
    combo = {("C", frozenset({"A", "B"})): 7.0}

    def fake_score_fn(node, parents, Z=None, **kwargs):
        pset = frozenset(parents)
        score = base[node]
        score += sum(w.get((p, node), 0.0) for p in pset)
        score += combo.get((node, pset), 0.0)
        return score

    return fake_score_fn, base, w, combo


@pytest.fixture
def fake_data():
    np.random.seed(42)
    return pd.DataFrame(
        np.random.random((1000, 4)),
        columns=["A", "B", "C", "D"],
    )


""" 1. Compatibility Tests """


def make_estimator():
    return TOPIC()


@parametrize_with_checks([make_estimator()])
def test_topic_compatibility(estimator, check):
    check(estimator)


""" 2. Unit Tests (fake score function) """


def test_unit_improvement_matrix():
    topic = TOPIC()
    candidates = list(["A", "B", "C"])
    dag = DAG()
    dag.add_nodes_from(candidates)
    dag.add_edge("A", "C")

    score_fn, base, w, combo = fake_score_fn_factory()
    topic.score_fn_ = score_fn

    mat = topic._improvement_matrix(candidates=candidates, dag_current=dag)

    assert mat.shape == (3, 3)
    assert np.all(np.diag(mat) == 0.0)

    assert set(dag.get_parents("C")) == {"A"}
    assert set(dag.get_parents("A")) == set()
    assert set(dag.get_parents("B")) == set()

    def expected(dag, score_fn, cause, effect):
        if cause == effect:
            return 0.0
        current_parents = list(dag.get_parents(effect))
        old = score_fn(effect, current_parents)
        new = score_fn(effect, current_parents + [cause])
        return new - old

    iB = candidates.index("B")
    iC = candidates.index("C")
    assert mat[iB, iC] == pytest.approx(expected(dag, score_fn, "B", "C"))

    iA = candidates.index("A")
    iB = candidates.index("B")
    assert mat[iA, iB] == pytest.approx(expected(dag, score_fn, "A", "B"))

    assert mat[iB, iA] == pytest.approx(expected(dag, score_fn, "B", "A"))

    for cause in candidates:
        for effect in candidates:
            ic = candidates.index(cause)
            ie = candidates.index(effect)
            assert mat[ic, ie] == pytest.approx(expected(dag, score_fn, cause, effect))


def test_unit_next_node_in_topological_order(monkeypatch):
    topic = TOPIC()
    candidates = ["A", "B", "C"]
    dag = DAG()
    dag.add_nodes_from(candidates)

    improv = np.array(
        [
            [0.0, 10.0, 2.0],
            [1.0, 0.0, 5.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=float,
    )

    def fake_improvement_matrix(cands, dag_current):
        assert cands == candidates  # sanity: same ordering
        return improv

    monkeypatch.setattr(topic, "_improvement_matrix", fake_improvement_matrix)

    # def _score_fn(node, parents):
    #    return 0.0

    topic.score_fn_ = lambda node, parents: 0.0  # _score_fn
    source, meta = topic._next_node_in_topological_order(
        candidates=candidates, dag_current=dag
    )

    assert source == "A"
    assert meta["source_idx"] == 0
    assert meta["candidates"] == candidates

    delta_expected = improv - improv.T
    np.fill_diagonal(delta_expected, -np.inf)

    assert np.allclose(np.array(meta["improvement_matrix"]), improv)

    delta_from_meta = np.array(meta["delta_matrix"], dtype=float)
    for node_i in range(len(candidates)):
        for node_j in range(len(candidates)):
            if node_i == node_j:
                assert meta["delta_matrix"][node_i][node_j] == -np.inf
            else:
                assert delta_from_meta[node_i, node_j] == pytest.approx(
                    delta_expected[node_i, node_j]
                )

    incoming_pressure = np.max(delta_expected, axis=0)
    order_idx_expected = list(np.argsort(incoming_pressure))

    ranking = meta["ranking"]
    assert [r["node"] for r in ranking] == [candidates[i] for i in order_idx_expected]
    assert ranking[0]["incoming_pressure"] == pytest.approx(
        float(incoming_pressure[order_idx_expected[0]])
    )


def test_next_node_in_topological_order_tie(monkeypatch):
    topic = TOPIC()
    candidates = ["A", "B"]
    dag = DAG()
    dag.add_nodes_from(candidates)

    improv = np.array([[0.0, 1.0], [1.0, 0.0]])
    monkeypatch.setattr(topic, "_improvement_matrix", lambda *args, **kwargs: improv)

    # def _score_fn(node, parents):
    #    return 0.0

    # topic.score_fn_ = _score_fn
    topic.score_fn_ = lambda node, parents: 0.0  # _score_fn
    source, meta = topic._next_node_in_topological_order(
        candidates=candidates,
        dag_current=dag,
    )
    assert source == "A"
    assert meta["source_idx"] == 0


def test_find_removable_edge_single_parent():
    topic = TOPIC()

    # def score_fn(child, parents):
    #    return 0.0

    # topic.score_fn_ = score_fn
    topic.score_fn_ = lambda node, parents: 0.0  # _score_fn
    removed_found, best_parent, best_harm, candidate_stats = topic._find_removable_edge(
        parents=["A"], child="X"
    )

    assert removed_found is False
    assert best_parent is None
    assert best_harm == float("inf")
    assert candidate_stats == []


def test_find_removable_edge_best_parent():
    topic = TOPIC()

    def score_fn(child, parents):
        s = set(parents)
        if s == {"A", "B", "C"}:
            return 30.0
        if s == {"B", "C"}:
            return 10.0
        if s == {"A", "C"}:
            return 25.0
        if s == {"A", "B"}:
            return 50.0

    topic.score_fn_ = score_fn
    removed_found, best_parent, best_harm, candidate_stats = topic._find_removable_edge(
        parents=["A", "B", "C"], child="X"
    )

    # harms: A=-20, B=-5, C=+20
    assert removed_found is True
    assert best_parent == "C"
    assert best_harm == pytest.approx(20.0)
    assert candidate_stats == [
        ("A", pytest.approx(-20.0)),
        ("B", pytest.approx(-5.0)),
        ("C", pytest.approx(20.0)),
    ]


def test_find_removable_edge_no_removable_candidate():
    topic = TOPIC()

    def score_fn(child, parents):
        return 100.0 - (3 - len(parents)) * 10.0

    topic.score_fn_ = score_fn
    removed_found, best_parent, best_harm, candidate_stats = topic._find_removable_edge(
        parents=["A", "B", "C"], child="X"
    )

    assert removed_found is False
    assert best_parent is None
    assert best_harm == 0.0
    assert candidate_stats == [
        ("A", pytest.approx(-10.0)),
        ("B", pytest.approx(-10.0)),
        ("C", pytest.approx(-10.0)),
    ]


def test_find_removable_edge_allows_small_negative_harm_due_to_float_noise():
    topic = TOPIC()

    def score_fn2(child, parents):
        if set(parents) == {"A", "B"}:
            return 1.0
        if set(parents) == {"B"}:
            return 1.0 - 1e-12

    topic.score_fn_ = score_fn2
    removed_found, best_parent, best_harm, _ = topic._find_removable_edge(
        parents=["A", "B"], child="X"
    )

    assert removed_found is True
    assert best_parent == "A"


def test_remove_ingoing_edges_iterative_removal(monkeypatch):
    topic = TOPIC()
    dag = DAG()
    dag.add_nodes_from(["A", "B", "C", "X"])
    dag.add_edges_from([("A", "X"), ("B", "X"), ("C", "X")])

    sequence = [
        (True, "B", -0.3, [("A", 0.1), ("B", -0.3), ("C", 0.2)]),
        (True, "A", -0.1, [("A", -0.1), ("C", 0.05)]),
        (False, None, float("inf"), []),
    ]
    calls = {"i": 0}

    def fake_find(parents, child):
        out = sequence[calls["i"]]
        calls["i"] += 1
        return out

    monkeypatch.setattr(topic, "_find_removable_edge", fake_find)

    pruned_edges, meta = topic._remove_ingoing_edges("X", dag)

    assert pruned_edges == [
        {"from": "B", "to": "X", "diff": pytest.approx(-0.3)},
        {"from": "A", "to": "X", "diff": pytest.approx(-0.1)},
    ]

    assert meta == [
        {"from": "A", "to": "X", "diff": pytest.approx(0.1)},
        {"from": "B", "to": "X", "diff": pytest.approx(-0.3)},
        {"from": "C", "to": "X", "diff": pytest.approx(0.2)},
        {"from": "A", "to": "X", "diff": pytest.approx(-0.1)},
        {"from": "C", "to": "X", "diff": pytest.approx(0.05)},
    ]

    assert ("B", "X") not in dag.edges()
    assert ("A", "X") not in dag.edges()
    assert ("C", "X") in dag.edges()


def test_remove_ingoing_edges_no_parents():
    model = TOPIC()
    dag = DAG()
    dag.add_nodes_from(["X", "A"])

    pruned_edges, meta = model._remove_ingoing_edges("X", dag)

    assert pruned_edges == []
    assert meta == []
    assert list(dag.get_parents("X")) == []


def test_remove_ingoing_edges_breaks_immediately(monkeypatch):
    topic = TOPIC()
    dag = DAG()
    dag.add_nodes_from(["A", "B", "X"])
    dag.add_edges_from([("A", "X"), ("B", "X")])

    def fake_find(parents, child):
        return False, None, float("inf"), [("A", 0.2), ("B", 0.1)]

    monkeypatch.setattr(topic, "_find_removable_edge", fake_find)

    pruned_edges, meta = topic._remove_ingoing_edges("X", dag)

    assert pruned_edges == []
    assert meta == [
        {"from": "A", "to": "X", "diff": pytest.approx(0.2)},
        {"from": "B", "to": "X", "diff": pytest.approx(0.1)},
    ]

    assert ("A", "X") in dag.edges()
    assert ("B", "X") in dag.edges()


def test_remove_ingoing_edges_calls_find_until_none(monkeypatch):
    topic = TOPIC()
    dag = DAG()
    dag.add_nodes_from(["A", "B", "C", "X"])
    dag.add_edges_from([("A", "X"), ("B", "X"), ("C", "X")])

    calls = {"n": 0}

    def fake_find(parents, child):
        calls["n"] += 1
        if calls["n"] <= 2:
            return True, parents[0], -1.0, [(p, 0.0) for p in parents]
        return False, None, float("inf"), []

    monkeypatch.setattr(topic, "_find_removable_edge", fake_find)

    pruned_edges, meta = topic._remove_ingoing_edges("X", dag)

    assert calls["n"] == 3
    assert len(pruned_edges) == 2
    assert ("C", "X") in dag.edges()


def test_score_significant():
    topic = TOPIC()
    topic._init_score(pd.DataFrame())

    assert not topic._score_significant(-1.0)
    assert not topic._score_significant(0.0)
    assert topic._score_significant(0.01)
    assert topic._score_significant(10.0)


def test_add_outgoing_edges_adds_only_significant_and_skips_self(monkeypatch):
    topic = TOPIC()
    dag = DAG()
    dag.add_nodes_from(["A", "B", "C"])

    gains = {"B": 2.0, "C": -1.0}
    monkeypatch.setattr(
        topic,
        "_addition_gain",
        lambda cause, effect, dag_current: gains[effect],
    )
    monkeypatch.setattr(topic, "_score_significant", lambda gain: gain > 0)

    added, all = topic._add_outgoing_edges(
        source="A", candidates=["A", "B", "C"], dag_current=dag
    )

    assert ("A", "A") not in dag.edges()
    assert ("A", "B") in dag.edges()
    assert ("A", "C") not in dag.edges()

    assert added == [{"from": "A", "to": "B", "gain": pytest.approx(2.0)}]
    assert all == [
        {"from": "A", "to": "B", "gain": pytest.approx(2.0), "significant": True},
        {"from": "A", "to": "C", "gain": pytest.approx(-1.0), "significant": False},
    ]


_ERR = (
    r"Score function not initialized\. Call _init_score\(data\) or fit\(data\) first\."
)


def test_score_checks_init_score():
    topic = TOPIC()
    with pytest.raises(ValueError, match=_ERR):
        topic._score("X", ["A", "B"])


def test_addition_gain_checks_init_score():
    topic = TOPIC()
    dag = DAG()
    dag.add_nodes_from(["A", "B"])

    with pytest.raises(ValueError, match=_ERR):
        topic._addition_gain(cause="A", effect="B", dag_current=dag)


def test_improvement_matrix_checks_init_score():
    topic = TOPIC()
    dag = DAG()
    dag.add_nodes_from(["A", "B"])

    with pytest.raises(ValueError, match=_ERR):
        topic._improvement_matrix(candidates=["A", "B"], dag_current=dag)


def test_add_outgoing_edges_checks_init_score():
    topic = TOPIC()
    dag = DAG()
    dag.add_nodes_from(["A", "B", "C"])

    with pytest.raises(ValueError, match=_ERR):
        topic._add_outgoing_edges(source="A", candidates=["B", "C"], dag_current=dag)


def test_find_removable_edge_checks_init_score():
    topic = TOPIC()
    with pytest.raises(ValueError, match=_ERR):
        topic._find_removable_edge(parents=["A", "B"], child="X")


def test_remove_ingoing_edges_checks_init_score(fake_data):
    with pytest.raises(ValueError):
        est = TOPIC(return_type="")
        _ = est.fit(fake_data)


def test_TOPIC_checks_return_type():
    topic = TOPIC()
    dag = DAG()
    dag.add_nodes_from(["A", "B", "X"])
    dag.add_edge("A", "X")
    dag.add_edge("B", "X")

    with pytest.raises(ValueError, match=_ERR):
        topic._remove_ingoing_edges(source="X", dag_current=dag)


""" 3. Smoke Test (fake data) """


def test_fit_scoring_methods(fake_data):
    est = TOPIC()
    dag = est.fit(fake_data)
    assert dag is not None
    assert est.n_features_in_ == fake_data.shape[1]
    assert len(est.feature_names_in_) == len(
        np.asarray(fake_data.columns, dtype=object)
    )


@pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
@pytest.mark.parametrize("show_progress", [True, False])
@pytest.mark.parametrize("return_type", ["dag", "pdag"])
def test_arguments(fake_data, scoring_method, show_progress, return_type):
    est = TOPIC(
        scoring_method=scoring_method,
        show_progress=show_progress,
        return_type=return_type,
    )
    _ = est.fit(fake_data)
