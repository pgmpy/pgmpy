import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery import LiNGAM


@pytest.fixture
def rand_data():
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.uniform(size=(100, 3)),
        columns=list("ABC"),
    )
    data["B"] = 2.0 * data["A"] + data["B"]
    data["C"] = -1.5 * data["B"] + data["C"]
    return data


@pytest.fixture
def rand_data2():
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.laplace(size=(200, 5)),
        columns=list("ABCDE"),
    )

    data["B"] = 1.2 * data["A"] + data["B"]
    data["C"] = -1.5 * data["A"] + data["C"]
    data["D"] = 0.8 * data["B"] + data["D"]
    data["E"] = -0.7 * data["C"] + data["E"]

    return data


def test_fit_rand(rand_data):
    from sklearn.decomposition import FastICA

    algo = LiNGAM(fast_ica=FastICA(random_state=42))
    algo.fit(rand_data)
    graph = algo.causal_graph_

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


def test_fit_rand2(rand_data2):
    from sklearn.decomposition import FastICA

    algo = LiNGAM(fast_ica=FastICA(random_state=42))
    algo.fit(rand_data2)

    graph = algo.causal_graph_

    assert graph.has_edge("A", "B")
    assert graph.has_edge("B", "D")
    assert graph.has_edge("C", "E")
    assert graph.has_edge("A", "C")

    assert not graph.has_edge("B", "A")
    assert not graph.has_edge("C", "A")
    assert not graph.has_edge("D", "B")
    assert not graph.has_edge("E", "C")

    assert not graph.has_edge("C", "B")
    assert not graph.has_edge("B", "C")


def test_fit_custom_fast_ica(rand_data):
    from sklearn.decomposition import FastICA

    custom_ica = FastICA(random_state=42, max_iter=500, tol=1e-3)
    algo = LiNGAM(fast_ica=custom_ica)
    algo.fit(rand_data)
    graph = algo.causal_graph_

    assert graph.has_edge("A", "B")
    assert graph.has_edge("B", "C")
    assert not graph.has_edge("B", "A")
    assert not graph.has_edge("C", "B")
    assert not graph.has_edge("A", "C")

    # Test adjacency matrix structure
    B = algo.adjacency_matrix_
    assert B.shape == (3, 3)
