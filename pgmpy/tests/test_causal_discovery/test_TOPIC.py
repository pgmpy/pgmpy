import unittest
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery.TOPIC import TOPIC

""" Utils """
def fake_score_fn_factory():
    base = {"A": 0.0, "B": 10.0, "C": 20.0}
    w = {("A", "B"): 1.0,("C", "B"): -2.0,("B", "C"): 3.0,("A", "C"): 4.0}
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

    mat = topic._improvement_matrix(candidates=candidates, dag_current=dag, score_fn=score_fn)

    assert mat.shape == (3, 3)
    assert np.all(np.diag(mat) == 0.0)

    assert set(dag.get_parents("C")) == {"A"}
    assert set(dag.get_parents("A")) == set()
    assert set(dag.get_parents("B")) == set()

    def expected(dag, score_fn, cause, effect):
        if cause == effect: return 0.0
        current_parents = list(dag.get_parents(effect))
        old = score_fn(effect, current_parents)
        new = score_fn(effect, current_parents + [cause])
        return old - new


    iB = candidates.index("B")
    iC = candidates.index("C")
    assert mat[iB, iC] == pytest.approx(expected(dag, score_fn,"B", "C"))

    iA = candidates.index("A")
    iB = candidates.index("B")
    assert mat[iA, iB] == pytest.approx(expected(dag, score_fn,"A", "B"))

    assert mat[iB, iA] == pytest.approx(expected(dag, score_fn,"B", "A"))

    for cause in candidates:
        for effect in candidates:
            ic = candidates.index(cause)
            ie = candidates.index(effect)
            assert mat[ic, ie] == pytest.approx(expected(dag, score_fn, cause, effect))



""" 3. Smoke Tests (fake data) """
@pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
def test_fit_scoring_methods(fake_data, scoring_method):
    est = TOPIC(scoring_method=scoring_method)
    dag = est.fit(fake_data)
    assert dag is not None