# The tests in this file are validated against the lingam package v1.12.2.
# Helper function to convert node indices to letters (used for validation only):
# def num_letter(edges):
#     def num_to_letter(n):
#         return chr(ord('A') + n)
#     return [(num_to_letter(u), num_to_letter(v)) for u, v in edges]

import numpy as np
import numpy.testing as np_test
import pandas as pd
import pytest

from pgmpy.causal_discovery import LiNGAM
from pgmpy.datasets import load_dataset


@pytest.fixture
def rand_data():
    """
    A -> B -> C
    """
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
    r"""
        / -> B -> D
    A -
        \ -> C -> E
    """
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
    r"""
                     F --
                   /      \
            B --> D        --> H --
          /         \     /          \
         /            G --            \
       A                                --> J
         \            F --            /
           \        /      \        /
            C --> E         --> I --
                    \      /
                      G --
    """

    rng = np.random.default_rng(42)

    data = pd.DataFrame(
        rng.laplace(size=(5000, 10)),
        columns=list("ABCDEFGHIJ"),
    )

    # Level 1
    data["B"] = 1.5 * data["A"] + data["B"]
    data["C"] = -1.2 * data["A"] + data["C"]

    # Level 2
    data["D"] = 0.8 * data["B"] + data["D"]
    data["E"] = -1.0 * data["C"] + data["E"]

    # Level 3
    data["F"] = 1.3 * data["D"] + 0.7 * data["E"] + data["F"]
    data["G"] = -0.9 * data["D"] + 0.5 * data["E"] + data["G"]

    # Level 4
    data["H"] = 0.5 * data["F"] - 1.1 * data["G"] + data["H"]
    data["I"] = 0.6 * data["F"] + 0.8 * data["G"] + data["I"]

    # Level 5
    data["J"] = -0.7 * data["H"] + 0.9 * data["I"] + data["J"]

    return data


@pytest.fixture
def mpg_data():
    data = load_dataset("auto_mpg").data.dropna()
    continuous_cols = ["displacement", "horsepower", "weight", "acceleration", "mpg"]
    return data[continuous_cols]


def test_fit_rand(rand_data):
    # model = lingam.ICALiNGAM(random_state=42, max_iter=1000)

    algo = LiNGAM(random_state=42)
    algo.fit(rand_data)
    graph = algo.causal_graph_

    # print(num_letter(networkx.convert_matrix.from_numpy_array(model.adjacency_matrix_).edges()))
    # [('A', 'B'), ('B', 'C')]
    assert set(graph.edges()) == {("A", "B"), ("B", "C")}

    # Test adjacency matrix structure -- Main Goal is to check the association paths should be blocked
    Adj_matrix = algo.adjacency_matrix_
    assert Adj_matrix.shape == (3, 3)

    # print(model.adjacency_matrix_[0, 2]) -> 0.0
    assert Adj_matrix.loc["A", "C"] == 0


def test_fit_rand2(rand_data2):
    # model = lingam.ICALiNGAM(random_state=42)

    algo = LiNGAM(random_state=42)
    algo.fit(rand_data2)
    graph = algo.causal_graph_

    # print(num_letter(networkx.convert_matrix.from_numpy_array(model.adjacency_matrix_).edges()))
    # [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E')]
    assert set(graph.edges()) == {("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")}

    # Test adjacency matrix structure -- Main Goal is to check the association paths should be blocked
    Adj_matrix = algo.adjacency_matrix_
    assert Adj_matrix.shape == (5, 5)

    # print(model.adjacency_matrix_[0, 3]) -> 0.0
    assert Adj_matrix.loc["A", "D"] == 0
    # print(model.adjacency_matrix_[0, 4]) -> 0.0
    assert Adj_matrix.loc["A", "E"] == 0
    # print(model.adjacency_matrix_[1, 2]) -> 0.0
    assert Adj_matrix.loc["B", "C"] == 0
    # print(model.adjacency_matrix_[3, 4]) -> 0.0
    assert Adj_matrix.loc["D", "E"] == 0
    # print(model.adjacency_matrix_[1, 4]) -> 0.0
    assert Adj_matrix.loc["B", "E"] == 0
    # print(model.adjacency_matrix_[2, 3]) -> 0.0
    assert Adj_matrix.loc["C", "D"] == 0


def test_large_lingam_data(large_lingam_data):
    # model = lingam.ICALiNGAM(random_state=42)

    algo = LiNGAM(random_state=42)
    algo.fit(large_lingam_data)
    graph = algo.causal_graph_

    # print(num_letter(networkx.convert_matrix.from_numpy_array(model.adjacency_matrix_).edges()))
    # [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('E', 'G'), ('F', 'H'),
    #  ('F', 'I'), ('G', 'H'), ('G', 'I'), ('H', 'J'), ('I', 'J')]
    assert set(graph.edges()) == {
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "E"),
        ("D", "F"),
        ("D", "G"),
        ("E", "F"),
        ("E", "G"),
        ("F", "H"),
        ("F", "I"),
        ("G", "H"),
        ("G", "I"),
        ("H", "J"),
        ("I", "J"),
    }

    # Test adjacency matrix structure
    Adj_matrix = algo.adjacency_matrix_
    assert Adj_matrix.shape == (10, 10)

    # print(model.adjacency_matrix_)
    test_matrix = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.50456931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.21074647, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.80754453, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.99998171, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.31147539, 0.6850421, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.89242808, 0.49331201, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.49770536, -1.0948292, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.60129612, 0.79999919, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.70158999, 0.89914671, 0.0],
        ]
    )

    # The test_matrix (adapted from the reference lingam package) stores edge weights in [target, source] orientation.
    # Transpose it to match pgmpy's standard graph adjacency convention of [source, target].
    np_test.assert_almost_equal(Adj_matrix.to_numpy(), test_matrix.T)


def test_mpg_data(mpg_data):
    algo = LiNGAM(random_state=42)
    algo.fit(mpg_data)
