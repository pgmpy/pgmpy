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

    mat = topic._improvement_matrix(
        candidates=candidates, dag_current=dag, score_fn=score_fn
    )

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
        return old - new

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


def test_next_node_in_topological_order_selects_min(monkeypatch):
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

    def fake_improvement_matrix(cands, dag_current, score_fn, **kwargs):
        assert cands == candidates  # sanity: same ordering
        return improv

    monkeypatch.setattr(topic, "_improvement_matrix", fake_improvement_matrix)

    source, meta = topic._next_node_in_topological_order(
        candidates=candidates,
        dag_current=dag,
        score_fn=lambda node, parents: 0.0,
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

    source, meta = topic._next_node_in_topological_order(
        candidates=candidates,
        dag_current=dag,
        score_fn=lambda node, parents: 0.0,
    )
    assert source == "A"
    assert meta["source_idx"] == 0


""" 3. Smoke Tests (fake data) """


@pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
def test_fit_scoring_methods(fake_data, scoring_method):
    est = TOPIC(scoring_method=scoring_method)
    dag = est.fit(fake_data)
    assert dag is not None
