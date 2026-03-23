import numpy as np
import numpy.testing as np_test
import pandas as pd
import pytest
from sklearn.decomposition import FastICA

from pgmpy.causal_discovery import LiNGAM


@pytest.fixture
def rand_data():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        rng.uniform(size=(100, 3)),
        columns=list("ABC"),
    )
    data["B"] = 2.0 * data["A"] + data["B"]
    data["C"] = -1.5 * data["B"] + data["C"]
    return data


@pytest.fixture
def rand_data2():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        rng.laplace(size=(1000, 5)),
        columns=list("ABCDE"),
    )

    data["B"] = 1.2 * data["A"] + data["B"]
    data["C"] = -1.5 * data["A"] + data["C"]
    data["D"] = 0.8 * data["B"] + data["D"]
    data["E"] = -0.7 * data["C"] + data["E"]

    return data


@pytest.fixture
def large_lingam_data():
    rng = np.random.default_rng(42)

    data = pd.DataFrame(
        rng.laplace(size=(5000, 10)),
        columns=list("ABCDEFGHIJ"),
    )

    # Level 1 dependencies
    data["B"] = 1.5 * data["A"] + data["B"]
    data["C"] = -1.2 * data["A"] + 0.5 * data["B"] + data["C"]

    # Level 2 dependencies
    data["D"] = 0.8 * data["B"] - 0.6 * data["C"] + data["D"]
    data["E"] = -1.0 * data["C"] + data["E"]

    # Level 3 dependencies
    data["F"] = 1.3 * data["D"] + 0.7 * data["E"] + data["F"]
    data["G"] = -0.9 * data["D"] + data["G"]

    # Level 4 dependencies
    data["H"] = 0.5 * data["F"] - 1.1 * data["G"] + data["H"]
    data["I"] = 0.6 * data["E"] + 0.8 * data["H"] + data["I"]

    # Final node with multiple parents
    data["J"] = -0.7 * data["F"] + 0.9 * data["I"] - 0.5 * data["C"] + data["J"]

    return data


def test_fit_rand(rand_data):
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
    arr = np.array([[0.0, 0.0, 0.0], [2.0321982696, 0.0, 0.0], [0.0, -1.4574545280, 0.0]])

    np_test.assert_array_almost_equal(B, arr)


def test_fit_rand2(rand_data2):
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

    # Test adjacency matrix structure
    B = algo.adjacency_matrix_

    assert B.shape == (5, 5)

    arr = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.19469001, 0.0, 0.0, 0.0, 0.0],
            [-1.51738062, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.82198363, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.65074361, 0.0, 0.0],
        ]
    )

    np_test.assert_array_almost_equal(B, arr, decimal=2)


def test_large_lingam_data(large_lingam_data):

    algo = LiNGAM(fast_ica=FastICA(random_state=42))
    algo.fit(large_lingam_data)
    graph = algo.causal_graph_

    assert graph.has_edge("A", "B")
    assert graph.has_edge("A", "C")
    assert graph.has_edge("B", "C")
    assert graph.has_edge("B", "D")
    assert graph.has_edge("C", "D")
    assert graph.has_edge("C", "E")
    assert graph.has_edge("D", "F")
    assert graph.has_edge("E", "F")
    assert graph.has_edge("D", "G")
    assert graph.has_edge("F", "H")
    assert graph.has_edge("G", "H")
    assert graph.has_edge("E", "I")
    assert graph.has_edge("H", "I")
    assert graph.has_edge("F", "J")
    assert graph.has_edge("I", "J")
    assert graph.has_edge("C", "J")

    assert not graph.has_edge("B", "A")
    assert not graph.has_edge("C", "A")
    assert not graph.has_edge("C", "B")
    assert not graph.has_edge("D", "B")
    assert not graph.has_edge("D", "C")
    assert not graph.has_edge("E", "C")
    assert not graph.has_edge("F", "D")
    assert not graph.has_edge("F", "E")
    assert not graph.has_edge("G", "D")
    assert not graph.has_edge("H", "F")
    assert not graph.has_edge("H", "G")
    assert not graph.has_edge("I", "E")
    assert not graph.has_edge("I", "H")
    assert not graph.has_edge("J", "F")
    assert not graph.has_edge("J", "I")
    assert not graph.has_edge("J", "C")

    # Test adjacency matrix structure
    B = algo.adjacency_matrix_
    assert B.shape == (10, 10)

    arr = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.50239738, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.19284925, 0.49601381, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.82013001, -0.6041246, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.00434477, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.32481944, 0.70584334, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.87022438, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.49830465, -1.10152681, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.600331, 0.0, 0.0, 0.81615371, 0.0, 0.0],
            [0.0, 0.0, -0.49005547, 0.0, 0.0, -0.73179177, 0.0, 0.0, 0.92003576, 0.0],
        ]
    )

    np_test.assert_array_almost_equal(B, arr, decimal=2)


def test_fit_custom_fast_ica(rand_data):
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
