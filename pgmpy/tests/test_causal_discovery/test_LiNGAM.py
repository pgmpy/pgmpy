import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery import LiNGAM
from pgmpy.estimators import ExpertKnowledge


@pytest.fixture
def rand_data():
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.uniform(size=(int(1e4), 3)),
        columns=list("ABC"),
    )
    data["B"] = 2.0 * data["A"] + data["B"]
    data["C"] = -1.5 * data["B"] + data["C"]
    return data


def test_fit_rand(rand_data):
    algo = LiNGAM(random_state=42, threshold=0.1)
    algo.fit(rand_data)
    graph = algo.causal_graph_
    # Check if the causal graph is correct
    assert graph.has_edge("A", "B")
    assert graph.has_edge("B", "C")
    assert not graph.has_edge("B", "A")
    assert not graph.has_edge("C", "B")
    assert not graph.has_edge("A", "C")

    # Test adjacency matrix structure
    B = algo.adjacency_matrix_
    assert B.shape == (3, 3)
    assert B[1, 0] > 1.5
    assert B[2, 1] < -1.0


def test_expert_knowledge_rand(rand_data):
    ek = ExpertKnowledge(forbidden_edges=[("A", "B")], required_edges=[("C", "A")])
    algo = LiNGAM(random_state=42, expert_knowledge=ek)
    algo.fit(rand_data)

    assert not algo.causal_graph_.has_edge("A", "B")
    assert algo.causal_graph_.has_edge("C", "A")
