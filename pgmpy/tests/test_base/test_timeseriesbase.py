import numpy as np
import pytest

from pgmpy.base import TimeseriesDAG
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
        assert _TimeSeriesCoreGraph(edge_list=[(("A", -3), ("B", 0), "->")]).tau_max == 3
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("B", 0), "->")], tau_max=4)
        assert graph.tau_max == 4
        assert graph.to_adjacency(encoding="tigramite").shape == (2, 2, 5)

        # declared: an edge beyond it is an error, not a silent widening
        with pytest.raises(ValueError, match=r"exceeds the maximum lag \(tau_max\)"):
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

        # a refused edge leaves the graph as it was, an inferred tau_max included
        graph = _TimeSeriesCoreGraph()
        with pytest.raises(ValueError, match="points backwards in time"):
            graph.add_edge(("A", -1), ("B", 0), "<-")
        check_graph_status(graph, 0, [], 0, [], {})

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

    def test_variables(self):

        assert sample_graph1().variables == ["A", "B"]

    def test_get_parents_and_get_children(self):

        graph = sample_graph1()

        # one node
        assert graph.get_parents(("B", 0)) == {("A", 0), ("A", -2)}
        assert graph.get_children(("A", -1)) == {("A", 0)}

        # several nodes
        assert graph.get_parents([("B", 0), ("A", 0)]) == {("A", 0), ("A", -2), ("A", -1)}
        assert graph.get_children([("A", -1), ("A", -2)]) == {("A", 0), ("B", 0)}
        assert graph.get_parents(["B", "A"]) == {("A", 0), ("A", -2), ("A", -1)}

        assert graph.get_parents(nodes=("B", 0)) == {("A", 0), ("A", -2)}
        assert graph.get_children(nodes=[("A", -1)]) == {("A", 0)}

        # a bare name is the variable at every time it is stored: same-time and lagged children together
        assert graph.get_children("A") == {("A", 0), ("B", 0)}
        assert graph.get_children(("A", 0)) == {("B", 0)}

        # a variable stored only as a lagged parent has no parents, and its children are those of its lagged node
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("B", 0), "->")])
        assert graph.get_parents("A") == set()
        assert graph.get_children("A") == {("B", 0)}

        # fails: a node that is not in the graph, and a variable that is not in the graph
        with pytest.raises(ValueError, match="not in graph"):
            graph.get_parents(("A", 0))
        with pytest.raises(ValueError, match="Variable 'Z' not found in the graph"):
            graph.get_children("Z")

    def test_get_ancestors(self):
        assert sample_graph1().get_ancestors(("B", 0)) == {("A", 0), ("A", -1), ("A", -2), ("B", 0)}

    def test_get_topological_order(self):
        graph = sample_graph1()
        order = graph.get_topological_order()
        position = {node: i for i, node in enumerate(order)}

        for u, v in graph.get_edges(data=False):
            assert position[u] < position[v]

    def test_get_ancestral_graph(self):

        graph = sample_graph1()
        ancestral = graph.get_ancestral_graph([("B", 0)])

        assert isinstance(ancestral, _TimeSeriesCoreGraph)
        assert sorted(ancestral.get_edges(lag_relative=True)) == sorted(graph.get_edges(lag_relative=True))

    def test_get_skeleton(self):
        graph = sample_graph1()

        assert set(graph.get_skeleton().nodes()) == set(graph.nodes())

    def test_is_collider(self):
        assert sample_graph1().is_collider(("A", 0), ("B", 0), ("A", -2))

    def test_get_subgraph(self):
        graph = _TimeSeriesCoreGraph(
            edge_list=[(("A", -2), ("B", 0), "->"), (("A", 0), ("B", 0), "->"), (("B", 0), ("C", 0), "->")]
        )
        subgraph = graph.get_subgraph([("A", 0), ("B", 0)])

        assert set(subgraph.nodes()) == {("A", 0), ("B", 0)}
        assert subgraph.get_edges(lag_relative=True) == [("A", "B", 0, "->")]

        subgraph = graph.get_subgraph(["A", "B"])
        assert set(subgraph.nodes()) == {("A", -2), ("A", 0), ("B", 0)}
        assert sorted(subgraph.get_edges(lag_relative=True)) == [("A", "B", 0, "->"), ("A", "B", 2, "->")]

        subgraph = graph.get_subgraph([("A", -2)])
        assert subgraph.tau_max == 2

    def test_copy(self):

        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -2), ("B", 0), "->")], tau_max=3, latents={("A", -2)})
        copied = graph.copy()

        assert type(copied) is _TimeSeriesCoreGraph
        assert copied == graph
        assert copied.latents == {("A", -2)}
        assert set(copied.nodes()) == set(graph.nodes())
        check_graph_status(copied, 2, ["A", "B"], 3, [("A", "B", 2, "->")], {"latents": [("A", -2)]})

        copied.add_edge(("B", 0), ("C", 0), "->")
        assert not graph.has_edge(("B", 0), ("C", 0))

        inferred = _TimeSeriesCoreGraph(edge_list=[(("A", -2), ("B", 0), "->")]).copy()
        inferred.add_edge(("A", -4), ("B", 0), "->")
        assert inferred.tau_max == 4

        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -3), ("B", 0), "->"), (("C", 0), ("B", 0), "->")])
        graph.remove_edge(("A", -3), ("B", 0), "->")
        assert graph.copy() == graph

    def test_eq_and_hash(self):
        edges = [(("A", -2), ("B", 0), "->")]
        same = _TimeSeriesCoreGraph(edge_list=edges, tau_max=3)
        also_same = _TimeSeriesCoreGraph(edge_list=edges, tau_max=3)
        other_tau = _TimeSeriesCoreGraph(edge_list=edges, tau_max=2)
        other_class = TimeseriesDAG(edge_list=edges, tau_max=3)

        assert same == also_same
        assert hash(same) == hash(also_same)

        assert same != other_tau
        assert hash(same) != hash(other_tau)
        assert same != other_class

    def test_normalize_edge_with_no_edge_type(self):
        """Test that `None` means "any edge type", rather than standing in for a real one."""
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("B", 0), "->")])

        assert graph.has_edge(("A", -1), ("B", 0))
        assert graph.has_edge(("B", 0), ("A", -1))
        assert graph.get_lag(("B", 0), ("A", -1)) == 1
        assert graph._normalize_edge(("B", 0), ("A", -1), None)[2] is None


class TestTimeseriesCoreGraphRoles:
    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            (("A", -1), [("A", -1)]),  # a (variable, time) pair marks that one node
            ("A", [("A", -1), ("A", 0)]),  # a bare name marks every offset the variable occupies
        ],
        ids=["node_pair", "bare_name"],
    )
    def test_with_role_target_shape(self, target, expected):
        """
        Test that the shape of the target decides whether one node or a whole variable is marked.
        """
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("A", 0), "->")], tau_max=2, latents={target})
        assert sorted(graph.latents) == expected

    def test_with_role(self):
        # a DynamicNode is a node, not a bare variable name, even though it is not a tuple
        from pgmpy.models.DynamicBayesianNetwork import DynamicNode

        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("A", 0), "->")], tau_max=2)
        graph.with_role(role="latents", variables=[DynamicNode("A", 0)], inplace=True)
        assert sorted(graph.latents) == [("A", 0)]

        # a list of nodes is a list, not one (variable, time) pair -- as `get_subgraph` passes
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("A", 0), "->")], tau_max=2)
        graph.with_role(role="latents", variables=[("A", -1), ("A", 0)], inplace=True)
        assert sorted(graph.latents) == [("A", -1), ("A", 0)]

        # fails: the inherited error survives the expansion, rather than the name doing nothing
        graph = _TimeSeriesCoreGraph(edge_list=[("A", "B", "->")])
        with pytest.raises(ValueError, match="not found in the graph"):
            graph.with_role(role="latents", variables={"Z"}, inplace=True)

    def test_with_role_integer_variable_names(self):
        """
        Test that a list of integer names is a list of variables, not one (variable, time) pair.
        """
        array = np.zeros((3, 3, 2), dtype=int)
        array[0, 1, 1] = array[1, 2, 1] = array[2, 0, 1] = 1
        graph = _TimeSeriesCoreGraph.from_adjacency(array)

        both_variables = [(0, -1), (0, 0), (2, -1), (2, 0)]
        assert sorted(graph.with_role(role="latents", variables=[2, 0]).latents) == both_variables
        assert sorted(graph.with_role(role="latents", variables={2, 0}).latents) == both_variables

        # a tuple stays a node: with integer names (2, 0) is ambiguous, and is read as the node
        assert sorted(graph.with_role(role="latents", variables=(2, 0)).latents) == [(2, 0)]

    def test_without_role(self):
        """
        Test that `without_role` clears a role named by variable, not only by node.
        """
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("A", 0), "->")], tau_max=2, latents={"A"})
        graph.without_role(role="latents", variables={"A"}, inplace=True)

        assert sorted(graph.latents) == []

    def test_role_is_not_inherited_by_later_nodes(self):
        """
        Test that a role marks the nodes present when it is set, as on every other pgmpy graph.
        """
        graph = _TimeSeriesCoreGraph(edge_list=[("A", "B", "->")], tau_max=3, latents={"A"})
        graph.add_edge(("A", -2), ("B", 0), "->")

        assert sorted(graph.latents) == [("A", 0)]

    def test_copy_keeps_the_marked_nodes(self):
        """
        Test that `copy` carries over the nodes a variable-level role expanded to.
        """
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("A", 0), "->")], tau_max=2, latents={"A"})

        assert sorted(graph.copy().latents) == [("A", -1), ("A", 0)]
        assert graph.copy() == graph


class TestTimeseriesCoreGraphAdjacency:
    def test_to_adjacency(self):
        graph = sample_graph2()

        # tigramite is the default, and the array is (N, N, tau_max + 1)
        array = graph.to_adjacency()
        assert isinstance(array, np.ndarray)
        assert array.shape == (2, 2, 2)

        # a contemporaneous link is written from both sides; a lagged one only forwards in time
        array = graph.to_adjacency(encoding="tigramite")
        assert array[0, 1, 0] == "-->"
        assert array[1, 0, 0] == "<--"
        assert array[0, 1, 1] == "-->"
        assert array[1, 0, 1] == ""

        # `variables` orders the first two axes
        reordered = graph.to_adjacency(encoding="tigramite", variables=["B", "A"])
        assert reordered[1, 0, 0] == "-->"

        # fails: an incomplete variable order
        with pytest.raises(ValueError, match="missing from the requested order"):
            graph.to_adjacency(encoding="tigramite", variables=["A"])

        # fails: a variable named twice
        with pytest.raises(ValueError, match="must be unique"):
            graph.to_adjacency(encoding="tigramite", variables=["A", "A", "B"])

        # fails: `nodelist` orders a 2-D matrix, so it is gone from the signature
        with pytest.raises(TypeError, match="nodelist"):
            graph.to_adjacency(nodelist=[("A", 0)])

    def test_from_adjacency(self):
        array = np.zeros((2, 2, 2), dtype=int)
        array[0, 1, 0] = array[0, 1, 1] = 1
        graph = _TimeSeriesCoreGraph.from_adjacency(array, variables=["A", "B"])

        check_graph_status(graph, 3, ["A", "B"], 1, [("A", "B", 0, "->"), ("A", "B", 1, "->")], {})

        # without `variables` the graph comes back index-keyed
        array = np.zeros((2, 2, 1), dtype=int)
        array[0, 1, 0] = 1
        assert _TimeSeriesCoreGraph.from_adjacency(array).get_edges(lag_relative=True) == [(0, 1, 0, "->")]

        # a variable with no links of its own still becomes a node
        array = np.zeros((3, 3, 1), dtype=int)
        array[0, 1, 0] = 1
        assert _TimeSeriesCoreGraph.from_adjacency(array, variables=["A", "B", "C"]).variables == ["A", "B", "C"]

        # fails: the first two axes must agree
        with pytest.raises(ValueError, match=r"shape \(N, N, tau_max \+ 1\)"):
            _TimeSeriesCoreGraph.from_adjacency(np.zeros((2, 3, 2), dtype=int), encoding="tigramite")

        # fails: with one encoding there is nothing to auto-detect, so a 2-D array is simply wrong
        with pytest.raises(ValueError, match="three-dimensional array"):
            _TimeSeriesCoreGraph.from_adjacency(np.zeros((3, 3), dtype=int))

        # fails: the names must cover the first axis exactly, and be unique
        array = np.zeros((2, 2, 1), dtype=int)
        with pytest.raises(ValueError, match="names for"):
            _TimeSeriesCoreGraph.from_adjacency(array, variables=["A"])
        with pytest.raises(ValueError, match="must be unique"):
            _TimeSeriesCoreGraph.from_adjacency(array, variables=["A", "A"])

        # fails: a lag-0 link must be written in both cells, the second the mirror of the first
        for upper, lower in [("", "-->"), ("-->", ""), ("-->", "-->")]:
            array = np.full((2, 2, 1), "", dtype="<U3")
            array[0, 1, 0], array[1, 0, 0] = upper, lower
            with pytest.raises(ValueError, match="do not describe the same link"):
                _TimeSeriesCoreGraph.from_adjacency(array, variables=["A", "B"])

        # fails: a numeric array marks presence only
        array = np.zeros((2, 2, 1), dtype=int)
        array[0, 1, 0] = 3
        with pytest.raises(ValueError, match="0/1 link marker"):
            _TimeSeriesCoreGraph.from_adjacency(array, variables=["A", "B"])

        # a float array of exactly 0 and 1 is read as numbers; anything else raises rather than truncating to 0
        array = np.zeros((2, 2, 2))
        array[0, 1, 1] = 1.0
        assert _TimeSeriesCoreGraph.from_adjacency(array, variables=["A", "B"]).get_edges(lag_relative=True) == [
            ("A", "B", 1, "->")
        ]
        array[0, 1, 1] = 0.5
        with pytest.raises(ValueError, match="0.5 is not a 0/1 link marker"):
            _TimeSeriesCoreGraph.from_adjacency(array, variables=["A", "B"])

        # an object array of strings, as pandas `.values` gives, is read as marks
        array = np.full((2, 2, 2), "", dtype=object)
        array[0, 1, 1] = "-->"
        assert _TimeSeriesCoreGraph.from_adjacency(array, variables=["A", "B"]).get_edges(lag_relative=True) == [
            ("A", "B", 1, "->")
        ]

        # fails: an object array mixing marks and numbers
        array = np.zeros((2, 2, 2), dtype=object)
        array[0, 1, 1] = "-->"
        with pytest.raises(ValueError, match="mixes string marks and numbers"):
            _TimeSeriesCoreGraph.from_adjacency(array, variables=["A", "B"])

    @pytest.mark.parametrize("encoding", TWO_D_ENCODINGS)
    @pytest.mark.parametrize("direction", ["to", "from"])
    def test_two_dimensional_encodings_are_refused(self, direction, encoding):
        """
        Test that both adjacency methods refuse every encoding without a lag axis.
        """
        graph = sample_graph2()

        with pytest.raises(ValueError, match="only supports encoding='tigramite'"):
            if direction == "to":
                graph.to_adjacency(encoding=encoding)
            else:
                _TimeSeriesCoreGraph.from_adjacency(graph.to_adjacency(), encoding=encoding)

    def test_from_adjacency_rejects_conflicting_mark(self):
        """
        Test that `x-x` is refused by every class, no pgmpy edge type expressing it.
        """
        array = np.full((2, 2, 1), "", dtype="<U3")
        array[0, 1, 0] = array[1, 0, 0] = "x-x"

        with pytest.raises(ValueError, match="_TimeSeriesCoreGraph can hold"):
            _TimeSeriesCoreGraph.from_adjacency(array, variables=["A", "B"])

    def test_from_adjacency_rejects_lagged_backwards_mark(self):
        """
        Test that a lagged link pointing backwards in time is refused.
        """
        array = np.full((2, 2, 2), "", dtype="<U3")
        array[0, 1, 1] = "<--"

        with pytest.raises(ValueError, match="backwards in time"):
            _TimeSeriesCoreGraph.from_adjacency(array, variables=["A", "B"])

    def test_adjacency_round_trip(self):
        """
        Test that a graph survives a round trip through its tigramite array.
        """
        # with the encoding named
        graph = sample_graph1()
        array = graph.to_adjacency(encoding="tigramite")
        restored = _TimeSeriesCoreGraph.from_adjacency(array, encoding="tigramite", variables=graph.variables)
        assert restored == graph

        # a discovery algorithm's array goes straight in: tigramite is the default, not a detected format
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("B", 0), "->")])
        assert _TimeSeriesCoreGraph.from_adjacency(graph.to_adjacency(), variables=graph.variables) == graph

        # the array carries no names, so without `variables` the graph comes back index-keyed
        indexed = _TimeSeriesCoreGraph.from_adjacency(graph.to_adjacency())
        assert indexed != graph
        assert indexed.variables == [0, 1]
