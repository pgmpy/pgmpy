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
        assert graph.to_adjacency(encoding="tigramite").shape == (2, 2, 5)

        # declared: an edge beyond it is an error, not a silent widening
        with pytest.raises(ValueError, match="exceeds tau_max"):
            _TimeSeriesCoreGraph(edge_list=[(("A", -5), ("B", 0), "->")], tau_max=2)

        # fails: not a non-negative integer
        with pytest.raises(TypeError, match="must be an integer"):
            _TimeSeriesCoreGraph(tau_max=1.5)
        with pytest.raises(TypeError, match="must be an integer"):
            _TimeSeriesCoreGraph(tau_max=True)
        with pytest.raises(ValueError, match="non-negative"):
            _TimeSeriesCoreGraph(tau_max=-1)

    @pytest.mark.parametrize(("u", "v"), EQUIVALENT_FRAMES)
    def test_add_edge_is_frame_independent(self, u, v):
        graph = _TimeSeriesCoreGraph(edge_list=[(u, v, "->")])

        assert set(graph.nodes()) == {("A", -1), ("B", 0)}
        check_graph_status(graph, 2, ["A", "B"], 1, [("A", "B", 1, "->")], {})

    def test_add_edge(self):
        # an edge written backwards is canonicalized, not rejected
        graph = _TimeSeriesCoreGraph(edge_list=[(("B", 0), ("A", -1), "<-")])
        check_graph_status(graph, 2, ["A", "B"], 1, [("A", "B", 1, "->")], {})

        # a backwards undirected link keeps its marks on the endpoints it was given
        graph = _TimeSeriesCoreGraph(edge_list=[(("B", 0), ("A", 0), "--")])
        assert graph.get_edges(lag_relative=True) in ([("B", "A", 0, "--")], [("A", "B", 0, "--")])

        # at lag 0 "<-" is just "->" written the other way round
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", 0), ("B", 0), "<-")])
        check_graph_status(graph, 2, ["A", "B"], 0, [("B", "A", 0, "->")], {})

        # a lagged "<>" marks latent confounding, not backwards causation, so it is allowed
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("B", 0), "<>")])
        check_graph_status(graph, 2, ["A", "B"], 1, [("A", "B", 1, "<>")], {})

        # fails: a lagged link with an arrowhead on the earlier endpoint alone makes the present cause the past
        for edge_type in ["<-", "<o"]:
            with pytest.raises(ValueError, match="points backwards in time"):
                _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("B", 0), edge_type)])

        # a self-link across lags is legal; within one slice it is not
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("A", 0), "->")])
        check_graph_status(graph, 2, ["A"], 1, [("A", "A", 1, "->")], {})
        with pytest.raises(ValueError, match="Nodes cannot be the same"):
            _TimeSeriesCoreGraph(edge_list=[(("A", 0), ("A", 0), "->")])

        # fails: the same link written in a shifted frame is the link that already exists
        graph = _TimeSeriesCoreGraph(tau_max=1)
        graph.add_edge(("A", -1), ("B", 0), "->")
        with pytest.raises(ValueError, match="already exists"):
            graph.add_edge(("A", 4), ("B", 5), "->")
        assert len(graph.get_edges(lag_relative=True)) == 1

    def test_add_node(self):
        graph = sample_graph1()
        graph.add_node(("C", 7))

        assert ("C", 0) in set(graph.nodes())
        assert graph.variables == ["A", "B", "C"]

    def test_remove_edge(self):
        graph = sample_graph1()
        graph.remove_edge(("A", 3), ("B", 5), "->")

        assert sorted(graph.get_edges(lag_relative=True)) == [("A", "A", 1, "->"), ("A", "B", 0, "->")]

    def test_has_edge(self):
        graph = sample_graph1()

        assert graph.has_edge(("A", 10), ("B", 12))

        assert graph.has_edge(("B", 0), ("A", 0))

        assert graph.has_edge(("A", 0), ("B", 0), "->")
        assert not graph.has_edge(("B", 0), ("A", 0), "->")

        assert not graph.has_edge(("A", 0), ("C", 0))

    def test_get_lag(self):

        graph = sample_graph1()

        assert graph.get_lag(("A", -2), ("B", 0)) == 2
        assert graph.get_lag(("A", 5), ("B", 7)) == 2

        # fails: no such link at that lag
        with pytest.raises(ValueError, match="No such edge"):
            graph.get_lag(("B", 0), ("B", -1))

    def test_get_edges(self):

        graph = sample_graph1()

        assert graph.get_edges(lag=0, lag_relative=True) == [("A", "B", 0, "->")]

        # lagged only
        lagged = graph.get_edges(lag=range(1, graph.tau_max + 1), lag_relative=True)
        assert sorted(lagged) == [("A", "A", 1, "->"), ("A", "B", 2, "->")]

    def test_get_parents_by_lag(self):
        graph = sample_graph1()

        assert graph.get_parents_by_lag("B") == {0: ["A"], 2: ["A"]}
        assert graph.get_parents_by_lag("A") == {1: ["A"]}

        # a variable stored only as a lagged parent is in the graph, with no parents of its own
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("B", 0), "->")])
        assert graph.variables == ["A", "B"]
        assert graph.get_parents_by_lag("A") == {}

        # fails: a variable that is not in the graph
        with pytest.raises(ValueError, match="not in the graph"):
            graph.get_parents_by_lag("Z")
