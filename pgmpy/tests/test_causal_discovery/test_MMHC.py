import numpy as np
import pandas as pd
import pytest

from pgmpy.base import UndirectedGraph
from pgmpy.causal_discovery import MMHC


@pytest.fixture
def discrete_data():
    np.random.seed(42)
    data = pd.DataFrame(np.random.randint(0, 2, size=(2500, 4)), columns=list("XYZW"))
    data["sum"] = data.sum(axis=1)
    return data


def test_fit_returns_self(discrete_data):
    est = MMHC()
    result = est.fit(discrete_data)
    assert isinstance(result, MMHC)


def test_model_stored_after_fit(discrete_data):
    est = MMHC()
    est.fit(discrete_data)
    assert hasattr(est, "model_")


def test_state_names_stored_after_fit(discrete_data):
    est = MMHC()
    est.fit(discrete_data)
    assert hasattr(est, "state_names_")
    assert set(est.state_names_.keys()) == set(discrete_data.columns)


def test_mmpc_returns_undirected_graph(discrete_data):
    est = MMHC()
    est.state_names_ = {col: discrete_data[col].unique().tolist() for col in discrete_data.columns}
    skel = est.mmpc(discrete_data)
    assert isinstance(skel, UndirectedGraph)


def test_custom_significance_level(discrete_data):
    est = MMHC(significance_level=0.05)
    est.fit(discrete_data)
    assert hasattr(est, "model_")


def test_custom_tabu_length(discrete_data):
    est = MMHC(tabu_length=5)
    est.fit(discrete_data)
    assert hasattr(est, "model_")


def test_fit_correct_edges(discrete_data):
    np.random.seed(42)
    est = MMHC(significance_level=0.001)
    est.fit(discrete_data)
    edges = {frozenset(e) for e in est.model_.edges()}
    assert frozenset({"X", "sum"}) in edges
    assert frozenset({"Y", "sum"}) in edges
    assert frozenset({"Z", "sum"}) in edges
    assert frozenset({"W", "sum"}) in edges
