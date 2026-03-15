import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery import ExhaustiveSearch


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return pd.DataFrame(np.random.randint(0, 3, size=(500, 3)), columns=list("ABC"))


def test_exhaustive_search_fit(sample_data):
    est = ExhaustiveSearch()
    est.fit(sample_data)
    assert hasattr(est, "causal_graph_")
    assert hasattr(est, "adjacency_matrix_")


def test_exhaustive_search_nodes(sample_data):
    est = ExhaustiveSearch()
    est.fit(sample_data)
    assert set(est.causal_graph_.nodes()) == set(sample_data.columns)


def test_exhaustive_search_adjacency_matrix(sample_data):
    est = ExhaustiveSearch()
    est.fit(sample_data)
    assert est.adjacency_matrix_.shape == (3, 3)


def test_exhaustive_search_is_dag(sample_data):
    import networkx as nx

    est = ExhaustiveSearch()
    est.fit(sample_data)
    assert nx.is_directed_acyclic_graph(est.causal_graph_)


def test_exhaustive_search_invalid_data():
    with pytest.raises(Exception):
        est = ExhaustiveSearch()
        est.fit("not a dataframe")
