import itertools

import numpy as np
import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import SID, get_metrics
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


# These matrices are compacted from PR #1927, which transcribed them from the reference R
# implementation. Cells (1, 0)[3, 7], (2, 0)[4, 10], and (2, 0)[6, 10] differ from the values in
# that PR: every cell below was re-derived by brute-force enumeration of all simple paths under the
# generalized adjustment criterion, and the three cells disagree with the PR but agree with the
# brute-force result, so the PR's values are the incorrect ones.
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


def test_sid_handles_graphs_without_edges():
    isolated = DAG()
    isolated.add_nodes_from(["A", "B", "C"])
    sid = SID()

    # Also covers the identity case on the public API; the random-DAG check below exercises it on
    # the matrix level.
    assert sid(isolated, isolated) == 0
    # A missing edge only costs the ordered pairs whose intervention distribution it changes.
    assert sid(DAG([("A", "B"), ("B", "C")]), isolated) == 3


def test_sid_counts_adjusting_for_a_mediator_as_incorrect():
    # Adjusting for a mediator biases the effect even though no back-door path is left open, so
    # this is the smallest graph exercising part (a) of the adjustment criterion on its own.
    true_graph = DAG([("A", "M"), ("M", "B")])
    est_graph = DAG([("M", "A")])
    est_graph.add_node("B")

    assert SID()(true_graph, est_graph) == 5


def test_sid_is_not_the_default_supervised_metric():
    # `CausalDiscovery.score` picks `get_metrics(requires_true_graph=True, is_default=True)[0]`,
    # so a second default would make the choice depend on class-name ordering.
    defaults = get_metrics(requires_true_graph=True, is_default=True)

    assert [metric.__name__ for metric in defaults] == ["SHD"]


def _descendants(graph, node):
    """Every node reachable from ``node`` along directed edges, including itself."""
    seen, stack = {node}, [node]
    while stack:
        for successor in np.flatnonzero(graph[stack.pop()]):
            if successor not in seen:
                seen.add(successor)
                stack.append(successor)
    return seen


def _path_is_blocked(graph, path, conditioned):
    """Textbook d-separation test applied to one explicit path."""
    for previous, current, following in zip(path, path[1:], path[2:], strict=False):
        if graph[previous, current] and graph[following, current]:
            if not _descendants(graph, current) & conditioned:
                return True
        elif current in conditioned:
            return True
    return False


def _has_open_non_causal_path(graph, source, target, conditioned):
    skeleton = graph | graph.T

    def walk(path, visited):
        if path[-1] == target:
            is_causal = all(graph[a, b] for a, b in zip(path, path[1:], strict=False))
            return not is_causal and not _path_is_blocked(graph, path, conditioned)
        return any(
            walk([*path, node], visited | {node}) for node in np.flatnonzero(skeleton[path[-1]]) if node not in visited
        )

    return walk([source], {source})


def _sid_matrix_by_enumeration(true_graph, est_graph):
    """Reference oracle: the adjustment criterion applied to every enumerated path.

    Exponential and independent of the algorithm under test, so it pins down the
    expected values without reusing any of :mod:`pgmpy.metrics.sid`.
    """
    n_nodes = true_graph.shape[0]
    incorrect = np.zeros((n_nodes, n_nodes), dtype=bool)

    for source, target in itertools.permutations(range(n_nodes), 2):
        conditioned = set(np.flatnonzero(est_graph[:, source]).tolist())
        if target in conditioned:
            # The estimated intervention distribution collapses to p(x_target),
            # which is right exactly when the source has no causal effect on it.
            incorrect[source, target] = target in _descendants(true_graph, source) - {source}
            continue

        on_causal_path = {
            node
            for node in range(n_nodes)
            if node != source and node in _descendants(true_graph, source) and target in _descendants(true_graph, node)
        }
        forbidden = set().union(*(_descendants(true_graph, node) for node in on_causal_path), set())
        incorrect[source, target] = bool(conditioned & forbidden) or _has_open_non_causal_path(
            true_graph, source, target, conditioned
        )

    return incorrect


def test_sid_matrix_matches_brute_force_enumeration_on_random_dags():
    rng = np.random.default_rng(0)

    for _ in range(100):
        n_nodes = int(rng.integers(3, 8))
        order = rng.permutation(n_nodes)
        true_graph = np.triu(rng.random((n_nodes, n_nodes)) < 0.45, 1)[order][:, order]
        est_graph = np.triu(rng.random((n_nodes, n_nodes)) < 0.45, 1)[order][:, order]

        np.testing.assert_array_equal(
            _sid_matrix(true_graph, est_graph),
            _sid_matrix_by_enumeration(true_graph, est_graph),
        )
        # A graph is always a perfect estimate of itself.
        assert not _sid_matrix(true_graph, true_graph).any()


@pytest.mark.parametrize(("true_index", "est_index"), REFERENCE_RESULTS)
def test_sid_matrix_matches_reference_and_brute_force(true_index, est_index):
    # The reference values were transcribed from the R implementation in PR #1927; the enumeration
    # re-derives them from the adjustment criterion. Asserting both against the same expectation is
    # what settles the three cells where the two sources disagree.
    expected = _matrix(REFERENCE_RESULTS[(true_index, est_index)])
    true_graph, est_graph = REFERENCE_GRAPHS[true_index], REFERENCE_GRAPHS[est_index]

    np.testing.assert_array_equal(_sid_matrix(true_graph, est_graph), expected)
    np.testing.assert_array_equal(_sid_matrix_by_enumeration(true_graph, est_graph), expected)
