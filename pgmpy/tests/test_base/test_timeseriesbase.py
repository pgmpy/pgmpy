import pytest

from pgmpy.base._timeseriesbase import _TimeSeriesCoreGraph


def sample_graph1():
    """
    Sample graph for testing the query and traversal methods of `_TimeSeriesCoreGraph`.
    Covers a self-link, a lag-2 link and a contemporaneous link into `B`.

    Notes
    -----
    Nodes are written as `(variable, time)` in the stored lag-relative frame,
    where the present slice is time 0 and parents sit at negative times.

        +---------+         +---------+         +---------+
        | (A, -2) |         | (A, -1) | ------> | (A,  0) |
        +---------+         +---------+         +---------+
             |                                       |
             |              +---------+              |
             +------------> | (B,  0) | <------------+
                            +---------+
    """
    edges = [
        (("A", -1), ("A", 0), "->"),
        (("A", -2), ("B", 0), "->"),
        (("A", 0), ("B", 0), "->"),
    ]
    return _TimeSeriesCoreGraph(edge_list=edges)


def sample_graph2():
    """
    Sample graph for testing the adjacency methods of `_TimeSeriesCoreGraph`.
    Two variables, `tau_max` of 1, with both a lagged and a contemporaneous link.

    Notes
    -----
        +---------+         +---------+
        | (A, -1) |         | (A,  0) |
        +---------+         +---------+
             |                   |
             |   +---------+     |
             +-> | (B,  0) | <---+
                 +---------+
    """
    edges = [
        (("A", -1), ("B", 0), "->"),
        (("A", 0), ("B", 0), "->"),
    ]
    return _TimeSeriesCoreGraph(edge_list=edges)


def check_graph_status(
    graph,
    node_count: int,
    variables: list,
    tau_max: int,
    links: list,
    roles: dict,
):
    """Common graph state checking function."""
    assert len(graph.nodes) == node_count
    assert graph.variables == variables
    assert graph.tau_max == tau_max
    assert sorted(graph.get_edges(lag_relative=True)) == sorted(links)
    assert graph.get_role_dict() == roles


TWO_D_ENCODINGS = ["edge_type", "binary", "bnlearn", "causal-learn", "pcalg"]
EQUIVALENT_FRAMES = [
    (("A", -1), ("B", 0)),
    (("A", 0), ("B", 1)),
    (("A", 7), ("B", 8)),
    (("A", -100), ("B", -99)),
]


class TestTimeseriesCoreGraph:
    def test_init(self):
        # empty graph
        graph = _TimeSeriesCoreGraph()
        check_graph_status(graph, 0, [], 0, [], {})

        # nodes from edges
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("B", 0), "->")])

        assert set(graph.nodes()) == {("A", -1), ("B", 0)}
        check_graph_status(graph, 2, ["A", "B"], 1, [("A", "B", 1, "->")], {})

        # a bare variable name is the present slice
        graph = _TimeSeriesCoreGraph(edge_list=[("A", "B", "->")])
        check_graph_status(graph, 2, ["A", "B"], 0, [("A", "B", 0, "->")], {})

        # roles are passed through to `_CoreGraph`
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("B", 0), "->")], latents={("A", -1)})

        assert graph.latents == {("A", -1)}
        check_graph_status(graph, 2, ["A", "B"], 1, [("A", "B", 1, "->")], {"latents": [("A", -1)]})

        # fails: a node that is not a (variable, time) pair
        with pytest.raises(ValueError, match=r"\(variable, time\) pair"):
            _TimeSeriesCoreGraph(edge_list=[(("A", 0, 1), ("B", 0), "->")])
        with pytest.raises(TypeError, match="must be an integer"):
            _TimeSeriesCoreGraph(edge_list=[(("A", "x"), ("B", 0), "->")])

    def test_tau_max(self):
        assert _TimeSeriesCoreGraph(edge_list=[(("A", -3), ("B", 0), "->")])._tau_max == 3
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("B", 0), "->")], tau_max=4)
        assert graph.tau_max == 4
