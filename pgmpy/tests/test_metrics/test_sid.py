import numpy as np
import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import SID
from pgmpy.metrics.sid import _compute_path_matrix, _sid_matrix
from pgmpy.models import DiscreteBayesianNetwork


def _matrix(rows):
    return np.array([[value == "1" for value in row] for row in rows])


REFERENCE_GRAPHS = (
    _matrix(
        (
            "00000001000",
            "00000001000",
            "00000101100",
            "10101111101",
            "11000100000",
            "10000000000",
            "11001001100",
            "00000000000",
            "01001101001",
            "10100001001",
            "10000101000",
        )
    ),
    _matrix(
        (
            "00000010100",
            "00000000100",
            "00000000100",
            "00000100000",
            "10010110101",
            "01000000100",
            "00000001000",
            "00000000000",
            "00000001000",
            "00010010101",
            "00010110100",
        )
    ),
    _matrix(
        (
            "00100011000",
            "00010010000",
            "00010100000",
            "00000000000",
            "00100001000",
            "00000000000",
            "00000000000",
            "00010110000",
            "10110011010",
            "01010111000",
            "00010100000",
        )
    ),
)


# These matrices are compacted from PR #1927. Cells (1, 0)[3, 7],
# (2, 0)[4, 10], and (2, 0)[6, 10] use the canonical R implementation's
# child propagation instead of the PR's parent-propagation values.
REFERENCE_RESULTS = {
    (0, 1): (
        "01110111111",
        "10111011111",
        "11011111111",
        "11101111101",
        "11110111111",
        "11100011110",
        "11111101101",
        "11111100011",
        "11011111001",
        "00000000000",
        "11110111100",
    ),
    (1, 0): (
        "00000011000",
        "10110101111",
        "00000000000",
        "11001111111",
        "11110111111",
        "11000011110",
        "11001101111",
        "00000000000",
        "11001100011",
        "00000000000",
        "11111111100",
    ),
    (2, 1): (
        "01110111110",
        "10111011111",
        "11011111110",
        "11100111100",
        "00000000000",
        "11100011110",
        "01110101100",
        "11111110010",
        "11110111010",
        "10110111100",
        "00000000000",
    ),
    (1, 2): (
        "01111111111",
        "10011111101",
        "01010111111",
        "11001111100",
        "00000000000",
        "11011011100",
        "00111101001",
        "01110110001",
        "11111111011",
        "11111111101",
        "11011111110",
    ),
    (0, 2): (
        "01111111011",
        "10111111101",
        "11011111111",
        "11101111101",
        "11110111111",
        "11011011100",
        "11111101101",
        "01110110001",
        "11111111011",
        "11111111101",
        "11111111110",
    ),
    (2, 0): (
        "01110111100",
        "10110111010",
        "11011111101",
        "11101111111",
        "11110111011",
        "11000011010",
        "11101101111",
        "00011010000",
        "11111111011",
        "10110111100",
        "11111111000",
    ),
}


@pytest.fixture
def paper_graphs():
    common_edges = [(source, target) for source in ("X1", "X2") for target in ("Y1", "Y2", "Y3")]
    true_graph = DAG(common_edges + [("X1", "X2")])
    supergraph = DAG(common_edges + [("X1", "X2"), ("Y1", "Y2")])
    reversed_graph = DAG(common_edges + [("X2", "X1")])
    return true_graph, supergraph, reversed_graph


def test_sid_paper_example(paper_graphs):
    true_graph, supergraph, reversed_graph = paper_graphs
    sid = SID()

    assert sid(true_graph, supergraph) == 0
    assert sid(true_graph, reversed_graph) == 8


def test_sid_is_asymmetric():
    empty_graph = DAG()
    empty_graph.add_nodes_from([0, 1])
    forward_graph = DAG([(0, 1)])
    reversed_graph = DAG([(1, 0)])
    sid = SID()

    assert sid(empty_graph, forward_graph) == 0
    assert sid(forward_graph, empty_graph) == 1
    assert sid(forward_graph, reversed_graph) == 2


def test_sid_supports_bayesian_networks_with_different_node_order():
    true_graph = DiscreteBayesianNetwork([(0, 1), (1, 2)])
    est_graph = DiscreteBayesianNetwork([(1, 2), (0, 1)])

    assert list(true_graph.nodes()) != list(est_graph.nodes())
    assert SID()(true_graph, est_graph) == 0


def test_sid_requires_dags_with_the_same_nodes():
    sid = SID()

    with pytest.raises(ValueError, match="same nodes"):
        sid(DAG([(0, 1)]), DAG([(0, 2)]))

    with pytest.raises(ValueError, match="must be one of"):
        sid(DAG([(0, 1)]), PDAG(edge_list=[(0, 1, "->")]))


def test_compute_path_matrix():
    graph = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]
    )

    np.testing.assert_array_equal(_compute_path_matrix(graph), np.triu(np.ones((4, 4), dtype=bool)))
    np.testing.assert_array_equal(_compute_path_matrix(np.empty((0, 0))), np.empty((0, 0), dtype=bool))


@pytest.mark.parametrize(("true_index", "est_index"), REFERENCE_RESULTS)
def test_sid_matrix_matches_r_reference(true_index, est_index):
    np.testing.assert_array_equal(
        _sid_matrix(REFERENCE_GRAPHS[true_index], REFERENCE_GRAPHS[est_index]),
        _matrix(REFERENCE_RESULTS[(true_index, est_index)]),
    )
