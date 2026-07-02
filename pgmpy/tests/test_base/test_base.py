#!/usr/bin/env python3

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import ADMG, DAG, MAG, PDAG
from pgmpy.base._base import _CoreGraph


def sample_graph1(edge_type=None):
    """
    Sample graph for testing node searching(`get_*`) method of `_CoreGraph` class.
    Tests node searching methods centered on node `C`.

    Notes
    -----
        +---+             +---+             +---+             +---+             +---+
        | A | [edge_type] | B | [edge_type] | C | [edge_type] | D | [edge_type] | E |
        +---+             +---+             +---+             +---+             +---+
    """
    edges = [
        ("A", "B", edge_type),
        ("B", "C", edge_type),
        ("C", "D", edge_type),
        ("D", "E", edge_type),
    ]
    return _CoreGraph(edge_list=edges)


def sample_graph2(edge_type=None):
    """
    Sample graph for testing node searching(`get_*`) method of `_CoreGraph` class.
    Tests node searching methods centered on node `B`.

    Notes
    -----
                                            +---+
                                [edge_type] | C |
        +---+             +---+             +---+
        | A | [edge_type] | B |
        +---+             +---+             +---+
                                [edge_type] | D |
                                            +---+


    """
    edges = [
        ("A", "B", edge_type),
        ("B", "C", edge_type),
        ("B", "D", edge_type),
    ]
    return _CoreGraph(edge_list=edges)


def sample_graph3():
    """
    sample graph for testing node searching(`get_*`) method of `_CoreGraph` class.

    Notes
    -----
    Used in `test_get_*_with_multiedges` method of `_CoreGraph` class.
    """
    edges = [
        ("A", "B", "->"),
        ("A", "C", "<-"),
        ("A", "D", "oo"),
        ("A", "E", "<>"),
        ("A", "F", "--"),
        ("A", "G", "-o"),
        ("A", "H", "o-"),
        ("A", "I", "o>"),
        ("A", "J", "<o"),
        ("B", "X", "->"),
        ("C", "Y", "<-"),
    ]
    return _CoreGraph(edge_list=edges)


def check_graph_status(
    graph,
    node_count: int,
    edge_count: int,
    exposures: set,
    outcomes: set,
    latents: set,
    roles: dict,
):
    """Common graph state checking function."""
    assert len(graph.nodes) == node_count
    assert len(graph.edges(keys=True, data=True)) == edge_count
    assert graph.exposures == exposures
    assert graph.outcomes == outcomes
    assert graph.latents == latents
    assert graph.get_role_dict() == roles


# All supported edge types, and the subset that are non-directed (everything except "->"/"<-").
EDGE_TYPES = ["--", "-o", "o-", "->", "<-", "o>", "<o", "<>", "oo"]
NON_DIRECTED = ["--", "-o", "o-", "o>", "<o", "<>", "oo"]


def assert_node_query_fails(method_name, empty_value):
    """Shared failure modes for the single-node `get_*` query methods of `_CoreGraph`."""
    with pytest.raises(ValueError):  # querying a node of an empty graph
        getattr(_CoreGraph(), method_name)("A")
    no_edges = _CoreGraph()
    no_edges.add_nodes_from(["A", "B"])
    assert getattr(no_edges, method_name)("A") == empty_value  # nodes but no edges
    check_graph_status(no_edges, 2, 0, set(), set(), set(), {})
    with pytest.raises(TypeError):  # missing required `node` argument
        getattr(_CoreGraph(), method_name)()
    with pytest.raises(ValueError):  # node not present in the graph
        getattr(_CoreGraph(), method_name)(1)


class TestCoreGraph:
    def test_init(self):
        """Test the initialization of a `_CoreGraph`."""
        # empty graph
        graph = _CoreGraph()
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # nodes from edges
        edges = [("A", "B", "->"), ("B", "C", "->")]
        graph = _CoreGraph(edge_list=edges)

        assert sorted(graph.nodes) == ["A", "B", "C"]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # edges with different edge types
        edges = [("A", "B", "--"), ("A", "B", "-o"), ("B", "C", "<>")]
        graph = _CoreGraph(edge_list=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": "-"}),
            ("A", "B", 1, {"A": "-", "B": "o"}),
            ("B", "C", 0, {"B": ">", "C": ">"}),
        ]
        check_graph_status(graph, 3, 3, set(), set(), set(), {})

        # exposures
        edges = [("A", "B", "->")]
        graph = _CoreGraph(edge_list=edges, exposures=["A"])

        assert sorted(graph.exposures) == ["A"]
        check_graph_status(graph, 2, 1, {"A"}, set(), set(), {"exposures": ["A"]})

        # outcomes
        edges = [("A", "B", "->")]
        graph = _CoreGraph(edge_list=edges, outcomes=["B"])

        assert sorted(graph.outcomes) == ["B"]
        check_graph_status(graph, 2, 1, set(), {"B"}, set(), {"outcomes": ["B"]})

        # latents
        edges = [("A", "B", "->")]
        graph = _CoreGraph(edge_list=edges, latents=["A"])

        assert sorted(graph.latents) == ["A"]
        check_graph_status(graph, 2, 1, set(), set(), {"A"}, {"latents": ["A"]})

        # custom roles
        edges = [("A", "B", "->")]
        graph = _CoreGraph(edge_list=edges, roles={"test_role": ["A"]})

        assert sorted(graph.get_roles()) == ["test_role"]
        check_graph_status(graph, 2, 1, set(), set(), set(), {"test_role": ["A"]})

        # all values together
        edges = [("A", "B", "->"), ("B", "C", "oo"), ("C", "D", "--")]
        graph = _CoreGraph(
            edge_list=edges,
            exposures=["A"],
            outcomes=["B"],
            latents=["C"],
            roles={"test_role": ["D"]},
        )

        check_graph_status(
            graph,
            4,
            3,
            {"A"},
            {"B"},
            {"C"},
            {
                "exposures": ["A"],
                "outcomes": ["B"],
                "latents": ["C"],
                "test_role": ["D"],
            },
        )

        # fails: invalid u/v (None), self-loop, and unsupported edge type
        invalid_edge_lists = [
            [("A", "B", "->"), (None, "A", "->"), ("B", "C", "->")],
            [("A", "B", "->"), ("A", None, "->"), ("B", "C", "->")],
            [("A", "A", "->")],
            [("A", "B", "->"), ("A", "A", "->"), ("C", "D", "--")],
            [("A", "B", "-->")],
            [("A", "B", "->"), ("A", "C", "o-->"), ("C", "D", "--")],
        ]
        for edge_list in invalid_edge_lists:
            with pytest.raises(ValueError):
                _CoreGraph(edge_list=edge_list)

        # wrong tuple length (2- or 4-tuple) reports "3 elements"
        with pytest.raises(ValueError, match="3 elements"):
            _CoreGraph(edge_list=[("A", "B")])
        with pytest.raises(ValueError, match="3 elements"):
            _CoreGraph(edge_list=[("A", "B", "key", "->")])

        # granting a role to a node not in the graph
        with pytest.raises(ValueError):
            _CoreGraph(roles={"test_role": "A"})
        with pytest.raises(ValueError):
            _CoreGraph(edge_list=[("A", "B", "->")], roles={"test_role1": "A", "test_role2": "C", "test_role3": "B"})

    def test_add_edge(self):
        """Test the `_CoreGraph.add_edge` method."""
        # directed edge
        graph = _CoreGraph()
        graph.add_edge("A", "C", "->")
        graph.add_edge("C", "B", "<-")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "C", 0, {"A": "-", "C": ">"}),
            ("C", "B", 0, {"B": "-", "C": ">"}),
        ]

        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # undirected edge
        graph = _CoreGraph()
        graph.add_edge("A", "C", "--")
        graph.add_edge("C", "B", "--")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("C", "B")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "C", 0, {"A": "-", "C": "-"}),
            ("C", "B", 0, {"B": "-", "C": "-"}),
        ]

        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # bidirected edge
        graph = _CoreGraph()
        graph.add_edge("A", "C", "<>")
        graph.add_edge("C", "B", "<>")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("C", "B")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "C", 0, {"A": ">", "C": ">"}),
            ("C", "B", 0, {"B": ">", "C": ">"}),
        ]

        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # unknown (circle) edges
        graph = _CoreGraph()
        graph.add_edge("A", "C", "-o")
        graph.add_edge("C", "B", "o-")
        graph.add_edge("D", "E", "o>")
        graph.add_edge("E", "F", "<o")
        graph.add_edge("G", "H", "oo")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("C", "B")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "C", 0, {"A": "-", "C": "o"}),
            ("C", "B", 0, {"B": "-", "C": "o"}),
            ("D", "E", 0, {"D": "o", "E": ">"}),
            ("E", "F", 0, {"F": "o", "E": ">"}),
            ("G", "H", 0, {"G": "o", "H": "o"}),
        ]

        check_graph_status(graph, 8, 5, set(), set(), set(), {})

        # multiedges of different types between the same pair
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        graph.add_edge("A", "B", "--")
        graph.add_edge("A", "B", "oo")

        assert graph.has_edge("A", "B")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": ">"}),
            ("A", "B", 1, {"A": ">", "B": ">"}),
            ("A", "B", 2, {"A": "-", "B": "-"}),
            ("A", "B", 3, {"A": "o", "B": "o"}),
        ]

        check_graph_status(graph, 2, 4, set(), set(), set(), {})

        # fails: duplicate edge of the same type is rejected
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        graph.add_edge("A", "B", "--")

        with pytest.raises(ValueError, match="already exists"):
            graph.add_edge("A", "B", "->")

        # fails: directed edge closing a cycle is rejected (`->` and `<-`)
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->")])

        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge("C", "A", "->")

        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge("A", "C", "<-")

        # fails: same node / unsupported edge_type / missing / None
        graph = _CoreGraph()
        with pytest.raises(ValueError):
            graph.add_edge("A", "A", "->")  # self-loop
        for bad_type in ["-->", "Invalid_value", 1, set(), dict()]:
            with pytest.raises(ValueError):
                graph.add_edge("A", "B", bad_type)
        with pytest.raises(TypeError):  # edge_type is required (no default)
            graph.add_edge("A", "B")
        with pytest.raises(ValueError, match=r"Got \('A', 'B', None\)"):  # None is not a valid edge_type
            graph.add_edge("A", "B", None)

        assert not graph.has_edge("A", "B")
        assert sorted(graph.edges(keys=True, data=True)) == []
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_add_edges_from(self):
        """Test the `_CoreGraph.add_edges_from` method."""
        # building a graph from an edge list, across edge types / mixes / multiedges
        for edges, expected, node_count, edge_count in [
            (
                [("A", "B", "->"), ("B", "C", "->")],
                [("A", "B", 0, {"A": "-", "B": ">"}), ("B", "C", 0, {"B": "-", "C": ">"})],
                3,
                2,
            ),
            (
                [("A", "B", "--"), ("B", "C", "--")],
                [("A", "B", 0, {"A": "-", "B": "-"}), ("B", "C", 0, {"B": "-", "C": "-"})],
                3,
                2,
            ),
            (
                [("A", "B", "<>"), ("B", "C", "<>")],
                [("A", "B", 0, {"A": ">", "B": ">"}), ("B", "C", 0, {"B": ">", "C": ">"})],
                3,
                2,
            ),
            (
                [("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")],
                [
                    ("A", "B", 0, {"A": "-", "B": "o"}),
                    ("B", "C", 0, {"B": "o", "C": "-"}),
                    ("C", "D", 0, {"C": "o", "D": "o"}),
                ],
                4,
                3,
            ),
            (
                [("A", "B", "->"), ("B", "C", "--"), ("C", "D", "<>")],
                [
                    ("A", "B", 0, {"A": "-", "B": ">"}),
                    ("B", "C", 0, {"B": "-", "C": "-"}),
                    ("C", "D", 0, {"C": ">", "D": ">"}),
                ],
                4,
                3,
            ),
            (
                [("A", "B", "->"), ("A", "B", "--"), ("A", "B", "oo")],
                [
                    ("A", "B", 0, {"A": "-", "B": ">"}),
                    ("A", "B", 1, {"A": "-", "B": "-"}),
                    ("A", "B", 2, {"A": "o", "B": "o"}),
                ],
                2,
                3,
            ),
        ]:
            graph = _CoreGraph()
            graph.add_edges_from(edge_list=edges)
            assert sorted(graph.edges(keys=True, data=True)) == expected
            check_graph_status(graph, node_count, edge_count, set(), set(), set(), {})

        # fails: None node / same node / invalid edge_type each raise and leave the graph empty
        bad_edge_lists = [
            [("A", "B", "->"), (None, "A", "->"), ("B", "C", "->")],
            [("A", "B", "->"), ("A", None, "->"), ("B", "C", "->")],
            [("A", "B", "->"), ("A", "A", "->"), ("B", "C", "->")],
            [("A", "B", "->"), ("B", "C", "-->"), ("C", "D", "->")],
        ]
        for edge_list in bad_edge_lists:
            graph = _CoreGraph()
            with pytest.raises(ValueError):
                graph.add_edges_from(edge_list=edge_list)
            check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # wrong tuple length (2- or 4-tuple) reports "3 elements"
        with pytest.raises(ValueError, match="3 elements"):
            _CoreGraph().add_edges_from(edge_list=[("A", "B", "->"), ("A", "C"), ("B", "C", "->")])
        with pytest.raises(ValueError, match="3 elements"):
            _CoreGraph().add_edges_from(edge_list=[("A", "B", "->"), ("A", "C", "key", "->"), ("B", "C", "->")])

        # fails: duplicate edge within a single call (consistent with `add_edge`)
        graph = _CoreGraph()
        with pytest.raises(ValueError, match="already exists"):
            graph.add_edges_from([("A", "B", "--"), ("A", "B", "--")])

    def test_remove_edge(self):
        """Test removing an edge of a `_CoreGraph`."""
        # removing every edge of a pair/chain leaves the graph edge-free (across all edge types)
        for edges, node_count in [
            ([("A", "B", "->"), ("B", "C", "<-")], 3),
            ([("A", "B", "--"), ("B", "C", "--")], 3),
            ([("A", "B", "<>"), ("B", "C", "<>")], 3),
            ([("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")], 4),
        ]:
            graph = _CoreGraph(edge_list=edges)
            for u, v, et in edges:
                graph.remove_edge(u, v, et)
            assert sorted(graph.edges(keys=True, data=True)) == []
            check_graph_status(graph, node_count, 0, set(), set(), set(), {})

        # multiedges
        edges = [("A", "B", "->"), ("A", "B", "<>"), ("A", "B", "--")]
        graph = _CoreGraph(edge_list=edges)

        graph.remove_edge("A", "B", "->")
        graph.remove_edge("A", "B", "--")

        assert not graph.has_edge("A", "B", "->")
        assert graph.has_edge("A", "B", "<>")
        assert not graph.has_edge("A", "B", "--")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 1, {"A": ">", "B": ">"}),
        ]

        check_graph_status(graph, 2, 1, set(), set(), set(), {})

        # requires explicit edge_type (no None 'remove all' shortcut)
        edges = [("A", "B", "->"), ("A", "B", "<>"), ("A", "B", "--")]
        graph = _CoreGraph(edge_list=edges)

        # edge_type is required (no default).
        with pytest.raises(TypeError):
            graph.remove_edge("A", "B")

        # None is not a valid edge_type, and the error reports the offending edge.
        with pytest.raises(ValueError, match=r"Got \('A', 'B', None\)"):
            graph.remove_edge("A", "B", None)

        # Parallel edges must be removed one explicit type at a time.
        graph.remove_edge("A", "B", "->")
        graph.remove_edge("A", "B", "<>")
        graph.remove_edge("A", "B", "--")
        assert not graph.has_edge("A", "B")
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        # fails: None node / same node / unsupported type each raise, leaving the graph unchanged
        edges = [("A", "B", "->"), ("B", "C", "->")]
        for bad_args in [(None, "C", "->"), ("B", None, "->"), ("B", "B", "->"), ("B", "C", "invalid_value")]:
            graph = _CoreGraph(edge_list=edges)
            graph.remove_edge("A", "B", "->")
            with pytest.raises(ValueError):
                graph.remove_edge(*bad_args)
            check_graph_status(graph, 3, 1, set(), set(), set(), {})

        # removing an edge type that isn't present between the nodes
        graph = _CoreGraph(edge_list=edges)
        with pytest.raises(ValueError, match="not in graph"):
            graph.remove_edge("A", "B", "<>")
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # no edge at all between the pair (or absent nodes) -> ValueError, not AttributeError
        graph = _CoreGraph(edge_list=edges)
        with pytest.raises(ValueError, match="not in graph"):
            graph.remove_edge("A", "C", "->")  # A and C present but non-adjacent
        with pytest.raises(ValueError, match="not in graph"):
            graph.remove_edge("X", "Y", "->")  # nodes absent from the graph

    def test_edge_list_accepts_generator(self):
        """edge_list given as a one-shot generator must not be silently exhausted by validation."""
        g = _CoreGraph(edge_list=(e for e in [("A", "B", "->"), ("B", "C", "->")]))
        assert set(g.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}
        g.add_edges_from(e for e in [("C", "D", "->")])
        assert ("C", "D", "->") in g.get_edges(data=True)
        g.remove_edges_from(e for e in [("A", "B", "->")])
        assert ("A", "B", "->") not in g.get_edges(data=True)

    def test_remove_edges_from(self):
        """Test removing edges of a `_CoreGraph`."""
        # removing every edge of a chain leaves the graph edge-free, across all edge types
        for edges, node_count in [
            ([("A", "B", "->"), ("B", "C", "<-")], 3),
            ([("A", "B", "--"), ("B", "C", "--")], 3),
            ([("A", "B", "<>"), ("B", "C", "<>")], 3),
            ([("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")], 4),
            ([("A", "B", "->"), ("B", "C", "--"), ("C", "D", "<o")], 4),
        ]:
            graph = _CoreGraph(edge_list=edges)
            graph.remove_edges_from(edge_list=edges)
            assert sorted(graph.edges(keys=True, data=True)) == []
            check_graph_status(graph, node_count, 0, set(), set(), set(), {})

        # multiedges: only the listed types are removed, the rest stay
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("A", "B", "<>"), ("A", "B", "--")])
        graph.remove_edges_from(edge_list=[("A", "B", "->"), ("A", "B", "--")])
        assert graph.has_edge("A", "B")
        assert not graph.has_edge("A", "B", "->")
        assert graph.has_edge("A", "B", "<>")
        assert not graph.has_edge("A", "B", "--")
        assert sorted(graph.edges(keys=True, data=True)) == [("A", "B", 1, {"A": ">", "B": ">"})]
        check_graph_status(graph, 2, 1, set(), set(), set(), {})

        # fails: None node / 2-tuple / 4-tuple / same node / unsupported edge_type each raise, graph unchanged
        edges = [("A", "B", "->"), ("B", "C", "->")]
        for bad_edges, match in [
            ([("A", "B", "->"), (None, "C", "->")], None),
            ([("A", "B", "->"), ("B", None, "->")], None),
            ([("A", "B", "->"), ("B", "C")], "3 elements"),
            ([("A", "B", "->"), ("B", "C", "key", "->")], "3 elements"),
            ([("A", "B", "->"), ("B", "B", "->")], None),
            ([("A", "B", "->"), ("B", "C", "invalid_value")], None),
        ]:
            graph = _CoreGraph(edge_list=edges)
            with pytest.raises(ValueError, match=match):
                graph.remove_edges_from(bad_edges)
            check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_copy(self):
        """Test the `copy` method of the `_CoreGraph` class."""
        nodes_only = _CoreGraph()
        nodes_only.add_nodes_from(["A", "B", "C"])
        individual = _CoreGraph()
        individual.add_edge("A", "C", "->")
        individual.add_edge("C", "B", "<-")

        # copy() returns an equal graph for every construction variant (empty, nodes, edges, roles, ...)
        variants = [
            _CoreGraph(),
            nodes_only,
            individual,
            _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]),
            _CoreGraph(
                edge_list=[("A", "B", "->"), ("A", "B", "<>"), ("B", "C", "->"), ("C", "D", "oo")],
                exposures=["A"],
                outcomes=["C"],
                latents=["D"],
            ),
            _CoreGraph(
                edge_list=[("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")], roles={"test_role": ["A", "B"]}
            ),
            _CoreGraph(
                edge_list=[("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")],
                exposures=["A"],
                outcomes=["C"],
                latents=["D"],
                roles={"test_role": ["B"]},
            ),
        ]
        for graph in variants:
            graph_copy = graph.copy()
            assert graph == graph_copy
            assert graph_copy == graph

        # roles are deep-copied: adding a role to a node that already has one does not leak to the original
        graph = _CoreGraph(edge_list=[("A", "B", "->")], exposures=["A"])
        graph_copy = graph.copy()
        graph_copy.with_role("outcomes", ["A"], inplace=True)
        assert graph_copy.outcomes == {"A"} and graph.outcomes == set()

        # fails: invalid copy argument type
        with pytest.raises(TypeError):
            _CoreGraph().copy("invalid_value")

    def test_get_neighbors(self):
        """Test `get_neighbors` method of the `_CoreGraph` class."""
        # neighbors ignore edge orientation: every edge type gives the same result
        for et in EDGE_TYPES:
            g1, g2 = sample_graph1(et), sample_graph2(et)
            assert g1.get_neighbors("C") == {"B", "D"}
            assert g2.get_neighbors("B") == {"A", "C", "D"}
            check_graph_status(g1, 5, 4, set(), set(), set(), {})
            check_graph_status(g2, 4, 3, set(), set(), set(), {})

        # multiedges
        graph = sample_graph3()
        assert graph.get_neighbors("A") == {"B", "C", "D", "E", "F", "G", "H", "I", "J"}
        assert graph.get_neighbors("B") == {"A", "X"}
        assert graph.get_neighbors("C") == {"A", "Y"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # filtering by edge_type values
        assert sample_graph1("->").get_neighbors("C", "->") == {"D"}
        assert sample_graph1("->").get_neighbors("C", "<-") == {"B"}
        assert sample_graph2("->").get_neighbors("B", "->") == {"C", "D"}
        assert sample_graph2("->").get_neighbors("B", "<-") == {"A"}
        assert sample_graph1("-o").get_neighbors("C", "o-") == {"B"}
        assert sample_graph2("-o").get_neighbors("B", "o-") == {"A"}
        assert graph.get_neighbors("A", "->") == {"B"}
        assert graph.get_neighbors("A", "<-") == {"C"}
        assert graph.get_neighbors("A", "<>") == {"E"}
        assert graph.get_neighbors("B", "->") == {"X"}
        assert graph.get_neighbors("B", "<-") == {"A"}
        assert graph.get_neighbors("C", "->") == {"A"}
        assert graph.get_neighbors("C", "<-") == {"Y"}
        assert graph.get_neighbors("C", "<>") == set()

        # a collection of edge types unions the matches (list or set); empty collection -> none
        assert graph.get_neighbors("A", {"->", "<-"}) == {"B", "C"}
        assert graph.get_neighbors("A", ["->", "<>"]) == {"B", "E"}
        assert graph.get_neighbors("A", set()) == set()

        # fails: no nodes / nodes but no edges / wrong node value / wrong edge_type
        with pytest.raises(ValueError):
            _CoreGraph().get_neighbors("A")
        no_edges = _CoreGraph()
        no_edges.add_nodes_from(["A", "B"])
        assert no_edges.get_neighbors("A") == set()
        check_graph_status(no_edges, 2, 0, set(), set(), set(), {})
        with pytest.raises(TypeError):
            _CoreGraph().get_neighbors()
        with pytest.raises(ValueError):
            _CoreGraph().get_neighbors(1)
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        with pytest.raises(ValueError):
            graph.get_neighbors("A", "wrong_edge")
        with pytest.raises(ValueError):
            graph.get_neighbors("A", {"->", "wrong_edge"})

    def test_get_parents(self):
        """Test `get_parents` method of the `_CoreGraph` class."""
        # parents follow incoming directed edges; non-directed edges have none
        assert sample_graph1("->").get_parents("C") == {"B"}
        assert sample_graph2("->").get_parents("B") == {"A"}
        assert sample_graph1("<-").get_parents("C") == {"D"}
        assert sample_graph2("<-").get_parents("B") == {"C", "D"}
        for et in NON_DIRECTED:
            assert sample_graph1(et).get_parents("C") == set()
            assert sample_graph2(et).get_parents("B") == set()

        # multiedges
        graph = sample_graph3()
        assert graph.get_parents("A") == {"C"}
        assert graph.get_parents("B") == {"A"}
        assert graph.get_parents("C") == {"Y"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: empty graph / nodes but no edges / missing arg / node not in graph
        assert_node_query_fails("get_parents", set())

    def test_get_children(self):
        """Test `get_children` method of the `_CoreGraph` class."""
        # children follow outgoing directed edges; non-directed edges have none
        assert sample_graph1("->").get_children("C") == {"D"}
        assert sample_graph2("->").get_children("B") == {"C", "D"}
        assert sample_graph1("<-").get_children("C") == {"B"}
        assert sample_graph2("<-").get_children("B") == {"A"}
        for et in NON_DIRECTED:
            assert sample_graph1(et).get_children("C") == set()
            assert sample_graph2(et).get_children("B") == set()

        # multiedges
        graph = sample_graph3()
        assert graph.get_children("A") == {"B"}
        assert graph.get_children("B") == {"X"}
        assert graph.get_children("Y") == {"C"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: empty graph / nodes but no edges / missing arg / node not in graph
        assert_node_query_fails("get_children", set())

    def test_get_spouses(self):
        """Test `get_spouses` method of the `_CoreGraph` class."""
        # spouses come only from bidirected edges; every other edge type gives none
        assert sample_graph1("<>").get_spouses("C") == {"B", "D"}
        assert sample_graph2("<>").get_spouses("B") == {"A", "C", "D"}
        for et in [t for t in EDGE_TYPES if t != "<>"]:
            assert sample_graph1(et).get_spouses("C") == set()
            assert sample_graph2(et).get_spouses("B") == set()

        # multiedges
        graph = sample_graph3()
        assert graph.get_spouses("A") == {"E"}
        assert graph.get_spouses("E") == {"A"}
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: empty graph / nodes but no edges / missing arg / node not in graph
        assert_node_query_fails("get_spouses", set())

    def test_get_parents_children_ancestors_iterable(self):
        """`get_parents`/`get_children`/`get_ancestors` accept a single node or an iterable (union)."""
        g = _CoreGraph(edge_list=[("A", "C", "->"), ("B", "C", "->"), ("C", "D", "->"), ("X", "Y", "->")])
        # single node -> set (unchanged)
        assert g.get_parents("C") == {"A", "B"}
        assert g.get_children("C") == {"D"}
        assert g.get_ancestors("C") == {"A", "B", "C"}
        # an iterable of nodes -> union (set)
        assert g.get_parents(["C", "D"]) == {"A", "B", "C"}
        assert g.get_children(["A", "B"]) == {"C"}
        assert g.get_ancestors(["C", "Y"]) == {"A", "B", "C", "X", "Y"}
        assert g.get_ancestors({"D"}) == {"A", "B", "C", "D"}  # a set is a collection too
        assert g.get_descendants(["C", "X"]) == {"C", "D", "X", "Y"}
        assert g.get_descendants({"A"}) == {"A", "C", "D"}  # a set is a collection too

    def test_get_ancestors(self):
        """Test `get_ancestors` method of the `_CoreGraph` class."""
        # ancestors follow incoming directed edges and include the node itself
        assert sample_graph1("->").get_ancestors("C") == {"A", "B", "C"}
        assert sample_graph2("->").get_ancestors("B") == {"A", "B"}
        assert sample_graph1("<-").get_ancestors("C") == {"C", "D", "E"}
        assert sample_graph2("<-").get_ancestors("B") == {"B", "C", "D"}
        for et in NON_DIRECTED:
            assert sample_graph1(et).get_ancestors("C") == {"C"}
            assert sample_graph2(et).get_ancestors("B") == {"B"}

        # multiedges
        graph = sample_graph3()
        assert graph.get_ancestors("A") == {"A", "C", "Y"}
        assert graph.get_ancestors("X") == {"X", "B", "A", "C", "Y"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: empty graph / nodes but no edges / missing arg / node not in graph
        assert_node_query_fails("get_ancestors", {"A"})

    def test_get_descendants(self):
        """Test `get_descendants` method of the `_CoreGraph` class."""
        # descendants follow outgoing directed edges and include the node itself
        assert sample_graph1("->").get_descendants("C") == {"C", "D", "E"}
        assert sample_graph2("->").get_descendants("B") == {"B", "C", "D"}
        assert sample_graph1("<-").get_descendants("C") == {"A", "B", "C"}
        assert sample_graph2("<-").get_descendants("B") == {"A", "B"}
        for et in NON_DIRECTED:
            assert sample_graph1(et).get_descendants("C") == {"C"}
            assert sample_graph2(et).get_descendants("B") == {"B"}

        # multiedges
        graph = sample_graph3()
        assert graph.get_descendants("A") == {"A", "B", "X"}
        assert graph.get_descendants("Y") == {"Y", "C", "A", "B", "X"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: empty graph / nodes but no edges / missing arg / node not in graph
        assert_node_query_fails("get_descendants", {"A"})

    def test_get_reachable_nodes(self):
        """Test `get_reachable_nodes` method of the `_CoreGraph` class."""
        # directed-style edges reach forward only; symmetric edges reach the whole connected graph
        for et in ["->", "<-", "o>", "<o", "-o", "o-"]:
            assert sample_graph1(et).get_reachable_nodes("C", et) == {"C", "D", "E"}
            assert sample_graph2(et).get_reachable_nodes("B", et) == {"B", "C", "D"}
        for et in ["--", "<>", "oo"]:
            assert sample_graph1(et).get_reachable_nodes("C", et) == {"A", "B", "C", "D", "E"}
            assert sample_graph2(et).get_reachable_nodes("B", et) == {"A", "B", "C", "D"}

        # multiedges: each edge type reaches only its own neighbourhood
        graph = sample_graph3()
        assert graph.get_reachable_nodes("A", "->") == {"A", "B", "X"}
        assert graph.get_reachable_nodes("A", "<-") == {"A", "C", "Y"}
        assert graph.get_reachable_nodes("A", "oo") == {"A", "D"}
        assert graph.get_reachable_nodes("A", "<>") == {"A", "E"}
        assert graph.get_reachable_nodes("A", "--") == {"A", "F"}
        assert graph.get_reachable_nodes("A", "o>") == {"A", "I"}
        assert graph.get_reachable_nodes("A", "<o") == {"A", "J"}
        # a collection of edge types follows any of them
        assert graph.get_reachable_nodes("A", {"->", "<-"}) == {"A", "B", "C", "X", "Y"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: no nodes / nodes without edges / wrong input values
        with pytest.raises(ValueError):
            _CoreGraph().get_reachable_nodes("A", "->")
        no_edges = _CoreGraph()
        no_edges.add_nodes_from(["A", "B"])
        assert no_edges.get_reachable_nodes("A", "->") == {"A"}
        check_graph_status(no_edges, 2, 0, set(), set(), set(), {})

    def test_get_district(self):
        """`get_district`: the bidirected-connected component, available on the `_CoreGraph` base."""
        g = _CoreGraph(edge_list=[("A", "B", "->"), ("A", "C", "<>"), ("C", "D", "<>"), ("E", "F", "->")])
        # transitive closure over bidirected edges, including the node itself
        assert g.get_district("A") == {"A", "C", "D"}
        assert g.get_district("E") == {"E"}  # no bidirected edge -> just the node
        # union over a collection; a list/set is a collection, str/int/tuple is a single node
        assert g.get_district(["A", "F"]) == {"A", "C", "D", "F"}
        # non-string single node is treated as one node (matches the other relation accessors)
        gi = _CoreGraph(edge_list=[(1, 2, "<>"), (2, 3, "<>")])
        assert gi.get_district(1) == {1, 2, 3}
        # missing node raises ValueError, like the other accessors
        with pytest.raises(ValueError, match="not in graph"):
            g.get_district("Z")
        with pytest.raises(TypeError):
            _CoreGraph().get_reachable_nodes()
        with pytest.raises(ValueError):
            _CoreGraph().get_reachable_nodes("A")

    def test_get_edges(self):
        # all edges with and without data
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "o-")
        graph.add_edge("A", "B", "<>")

        assert sorted(graph.get_edges(data=False)) == sorted(
            [
                ("A", "B"),
                ("A", "B"),
                ("B", "C"),
            ]
        )
        assert sorted(graph.get_edges(data=True)) == sorted(
            [
                ("A", "B", "->"),
                ("A", "B", "<>"),
                ("B", "C", "o-"),
            ]
        )

        # reversed forms are returned in canonical orientation by flipping the node order:
        # "<-" -> "->", "-o" -> "o-", "<o" -> "o>"
        graph = _CoreGraph()
        graph.add_edge("C", "B", "<-")  # really B -> C
        graph.add_edge("E", "D", "-o")  # really D o- E
        graph.add_edge("G", "F", "<o")  # really F o> G
        assert set(graph.get_edges(data=True)) == {("B", "C", "->"), ("D", "E", "o-"), ("F", "G", "o>")}

        # edge_types filters the edges, matching on the canonical type ("->" also catches "<-")
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("C", "B", "<-"), ("B", "D", "<>"), ("D", "E", "--")])
        assert set(graph.get_edges(data=True, edge_types={"->"})) == {("A", "B", "->"), ("B", "C", "->")}
        assert set(graph.get_edges(data=True, edge_types={"<>", "--"})) == {("B", "D", "<>"), ("D", "E", "--")}
        assert set(graph.get_edges(data=False, edge_types={"->"})) == {("A", "B"), ("B", "C")}

    def test_get_edge_type(self):
        # edge retrieval: directed, bidirected, partially-directed, circle, and reversed lookups
        graph = _CoreGraph()

        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "--")
        graph.add_edge("B", "C", "<>")
        graph.add_edge("C", "D", "-o")
        graph.add_edge("D", "E", "<-")
        graph.add_edge("E", "F", "oo")
        graph.add_edge("F", "G", "o-")
        graph.add_node("H")

        assert graph.get_edge_type("A", "B") == {"->", "--"}
        assert graph.get_edge_type("B", "C") == {"<>"}
        assert graph.get_edge_type("D", "E") == {"<-"}
        assert graph.get_edge_type("E", "D") == {"->"}
        assert graph.get_edge_type("E", "F") == {"oo"}
        assert graph.get_edge_type("F", "G") == {"o-"}

        # fails: no edge exists between the two nodes
        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        with pytest.raises(ValueError):
            graph.get_edge_type("A", "B")

    def test_get_unique_edge_types(self):
        """Test `_CoreGraph.get_unique_edge_types`: distinct types, orientation-reversed forms collapsed."""
        assert _CoreGraph().get_unique_edge_types() == set()

        # ->/<-, o-/-o and o>/<o are orientation-reversed views of one edge type -> a single canonical form
        graph = _CoreGraph(
            edge_list=[
                ("A", "B", "->"),
                ("D", "C", "<-"),  # both directed -> "->"
                ("E", "F", "o-"),
                ("H", "G", "-o"),  # both circle-tail -> "o-"
                ("I", "J", "o>"),
                ("L", "K", "<o"),  # both circle-arrow -> "o>"
            ]
        )
        assert graph.get_unique_edge_types() == {"->", "o-", "o>"}

        # symmetric types and coincident edges contribute their (canonical) types
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        graph.add_edge("C", "D", "--")
        assert graph.get_unique_edge_types() == {"->", "<>", "--"}

    def test_has_edge(self):
        # has_edge: directed, undirected, bidirected, reversed forms, and missing edges
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "--")
        graph.add_edge("C", "D", "<>")
        graph.add_node("E")

        assert graph.has_edge("A", "B") is True
        assert graph.has_edge("A", "B", "->") is True
        assert graph.has_edge("B", "C", "--") is True
        assert graph.has_edge("C", "D", "<>") is True
        assert graph.has_edge("D", "C", "<>") is True
        assert graph.has_edge("B", "A", "<-") is True
        assert graph.has_edge("A", "B", "<-") is False
        assert graph.has_edge("A", "B", "--") is False
        assert graph.has_edge("A", "B", "<>") is False
        assert graph.has_edge("A", "B", "-o") is False
        assert graph.has_edge("A", "E") is False
        assert graph.has_edge("A", "E", "->") is False

        # fails: unsupported edge type
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "--")
        graph.add_edge("C", "D", "<>")
        graph.add_node("E")

        with pytest.raises(ValueError):
            graph.has_edge("A", "B", "invalid_edge_type")

    def test_replace_edge(self):
        # replace directed/bidirected/circle edge types in sequence
        graph = _CoreGraph()
        graph.add_edge("A", "B", "--")

        graph.replace_edge("A", "B", old_type="--", new_type="->")
        assert graph.get_edges(data=True) == (
            [
                ("A", "B", "->"),
            ]
        )

        graph.replace_edge("A", "B", old_type="->", new_type="<>")
        assert graph.get_edges(data=True) == (
            [
                ("A", "B", "<>"),
            ]
        )

        graph.replace_edge("A", "B", old_type="<>", new_type="oo")
        assert graph.get_edges(data=True) == (
            [
                ("A", "B", "oo"),
            ]
        )

        # fails: invalid old_type / invalid new_type / nonexistent edge
        graph = _CoreGraph()
        graph.add_node("C")
        graph.add_edge("A", "B", "--")

        with pytest.raises(ValueError):
            graph.replace_edge("A", "B", old_type="!!", new_type="->")

        with pytest.raises(ValueError):
            graph.replace_edge("A", "B", old_type="--", new_type="~~")

        with pytest.raises(ValueError):
            graph.replace_edge("B", "C", old_type="--", new_type="->")

        # a failed replace must not destroy the edge (atomicity): "->" here would close a cycle
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->"), ("C", "A", "--")])
        with pytest.raises(ValueError, match="cycle"):
            graph.replace_edge("C", "A", old_type="--", new_type="->")
        assert graph.has_edge("C", "A", "--")

    def test_get_marker(self):
        """`get_marker(u, v)` returns the endpoint marker at `v` (the second argument)."""
        graph = _CoreGraph(edge_list=[("A", "B", "o>"), ("B", "C", "->")])
        assert graph.get_marker("A", "B") == ">"
        assert graph.get_marker("B", "A") == "o"
        assert graph.get_marker("B", "C") == ">"
        assert graph.get_marker("C", "B") == "-"

        # fails: no edge between the nodes / ambiguous parallel edges
        with pytest.raises(ValueError, match="not in graph"):
            graph.get_marker("A", "C")
        graph.add_edge("A", "B", "<>")
        with pytest.raises(ValueError, match="parallel"):
            graph.get_marker("A", "B")

    def test_set_marker(self):
        """`set_marker(u, v, marker)` re-orients the single endpoint at `v`, leaving the other
        endpoint untouched -- the core orientation move of FCI-style algorithms."""
        graph = _CoreGraph(edge_list=[("A", "B", "oo")])
        graph.set_marker("A", "B", ">")  # A oo B  =>  A o> B
        assert graph.get_edges(data=True) == [("A", "B", "o>")]
        graph.set_marker("B", "A", "-")  # A o> B  =>  A -> B
        assert graph.get_edges(data=True) == [("A", "B", "->")]
        graph.set_marker("B", "A", ">")  # A -> B  =>  A <> B
        assert graph.get_edges(data=True) == [("A", "B", "<>")]

        # setting the marker it already has is a no-op
        graph.set_marker("A", "B", ">")
        assert graph.get_edges(data=True) == [("A", "B", "<>")]

        # subclass edge-type restrictions are enforced: a PDAG cannot hold "<>"
        pdag = PDAG(edge_list=[("A", "B", "->")])
        with pytest.raises(ValueError):
            pdag.set_marker("B", "A", ">")
        assert pdag.has_edge("A", "B", "->")

        # acyclicity is enforced, and a failed set_marker leaves the edge unchanged
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->"), ("C", "A", "--")])
        with pytest.raises(ValueError, match="cycle"):
            graph.set_marker("C", "A", ">")  # C -- A  =>  C -> A closes the cycle
        assert graph.has_edge("C", "A", "--")

        # fails: invalid marker / no edge between the nodes
        with pytest.raises(ValueError, match="marker"):
            graph.set_marker("A", "B", "x")
        with pytest.raises(ValueError, match="not in graph"):
            graph.set_marker("A", "Z", ">")

    def test_eq(self):
        """Test the `__eq__` method of the `_CoreGraph` class."""
        # empty graphs are equal
        graph = _CoreGraph()
        other = _CoreGraph()

        assert graph.__eq__(other) == True
        assert other.__eq__(graph) == True

        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # graphs with identical values are equal
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]
        roles = {"test_role": ["B"]}
        graph = _CoreGraph(
            edge_list=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

        other = _CoreGraph()
        other.add_edges_from(edge_list=edges)
        other.exposures = exposures
        other.outcomes = outcomes
        other.latents = latents
        other.with_role("test_role", ["B"], inplace=True)

        assert graph.__eq__(other) == True
        assert other.__eq__(graph) == True

        check_graph_status(
            graph,
            4,
            3,
            {"A"},
            {"C"},
            {"D"},
            {
                "exposures": ["A"],
                "outcomes": ["C"],
                "latents": ["D"],
                "test_role": ["B"],
            },
        )

        # fails: no argument / too many arguments / called on non-graph
        graph = _CoreGraph()
        with pytest.raises(TypeError):
            graph.__eq__()
        with pytest.raises(TypeError):
            graph.__eq__(_CoreGraph(), _CoreGraph())

        with pytest.raises(AssertionError):
            # Not a graph class
            other_str = "not a graph"
            assert other_str.__eq__(graph) == False

        # different graphs are not equal (class / edges / roles / non-graph)
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]
        roles = {"test_role": ["B"]}
        graph = _CoreGraph(
            edge_list=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

        # a different graph class is never equal (compared both directions)
        for other in [DAG(), ADMG(), PDAG()]:
            assert graph.__eq__(other) == False
            assert other.__eq__(graph) == False

        # Different edge_list
        other_edge_list = _CoreGraph(
            edge_list=[
                ("A", "B", "->"),
                ("B", "C", "<>"),
                ("B", "C", "->"),
                ("C", "D", "oo"),
            ],
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )
        assert graph.__eq__(other_edge_list) == False
        assert other_edge_list.__eq__(graph) == False

        # changing any single field (exposures/outcomes/latents/roles) breaks equality
        base = dict(edge_list=edges, exposures=exposures, outcomes=outcomes, latents=latents, roles=roles)
        for override in [
            {"exposures": ["B"]},
            {"outcomes": ["A"]},
            {"latents": ["B"]},
            {"roles": {"new_role": ["A"]}},
        ]:
            other = _CoreGraph(**{**base, **override})
            assert graph.__eq__(other) == False
            assert other.__eq__(graph) == False

        # Not a graph class
        other_str = "not a graph"
        assert graph.__eq__(other_str) == False

        # parallel edges added in a different order are still equal (and hash equally)
        g1 = _CoreGraph(edge_list=[("X", "Y", "->"), ("X", "Y", "<>")])
        g2 = _CoreGraph(edge_list=[("X", "Y", "<>"), ("X", "Y", "->")])
        assert g1 == g2
        assert hash(g1) == hash(g2)

        # identical structure but different class is never equal, in either direction -- the class
        # is part of the hash, so a subclass instance must not compare equal to its parent class
        cg = _CoreGraph(edge_list=[("A", "B", "->")])
        admg = ADMG(edge_list=[("A", "B", "->")])
        assert cg.__eq__(admg) == False
        assert admg.__eq__(cg) == False

    def test_hash(self):
        """Test `__hash__`: equal graphs hash equally and are usable in sets / as dict keys."""
        # equal graphs hash equally -- even when built with reversed edge orientations
        g1 = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>")], latents=["A"])
        g2 = _CoreGraph(edge_list=[("C", "B", "<>"), ("B", "A", "<-")], latents=["A"])
        assert g1 == g2
        assert hash(g1) == hash(g2)

        # usable as set members / dict keys: equal graphs collapse, distinct graphs stay separate
        g3 = _CoreGraph(edge_list=[("A", "B", "->")])  # different structure
        g4 = _CoreGraph(edge_list=[("A", "B", "->")], latents=["A"])  # same edges, different roles
        assert g3 != g4
        assert len({g1, g2, g3, g4}) == 3  # g1 == g2 collapse to one; g3 and g4 are distinct
        assert {g1: "x"}[g2] == "x"  # an equal graph retrieves the stored value

        # the class is part of the hash: identical structure but a different graph type hashes differently
        assert hash(ADMG(edge_list=[("A", "B", "->")])) != hash(MAG(edge_list=[("A", "B", "->")]))
        assert hash(_CoreGraph(edge_list=[("A", "B", "->")])) != hash(ADMG(edge_list=[("A", "B", "->")]))

    def test_to_markers(self):
        graph = _CoreGraph()
        # edge_type -> per-endpoint markers for the pair (A, B)
        cases = {
            "o-": {"A": "o", "B": "-"},
            "<o": {"A": ">", "B": "o"},
            "<>": {"A": ">", "B": ">"},
            "--": {"A": "-", "B": "-"},
            "->": {"A": "-", "B": ">"},
        }
        for edge_type, markers in cases.items():
            assert graph._to_markers(("A", "B", edge_type)) == markers

        # fails: invalid edge tuple with extra element
        with pytest.raises(ValueError):
            graph._to_markers(("A", "B", "key", "<-"))

    def test_to_edge_type(self):
        graph = _CoreGraph()
        # (u, v, markers) -> edge_type code: reverse / circle-line / arrow-circle / bidirected / directed / undirected
        cases = [
            ("B", "A", {"B": ">", "A": "-"}, "<-"),
            ("A", "B", {"A": "o", "B": "-"}, "o-"),
            ("A", "B", {"A": ">", "B": "o"}, "<o"),
            ("A", "B", {"A": ">", "B": ">"}, "<>"),
            ("A", "B", {"A": "-", "B": ">"}, "->"),
            ("A", "B", {"A": "-", "B": "-"}, "--"),
        ]
        for u, v, markers, edge_type in cases:
            assert graph._to_edge_type(u, v, markers) == edge_type

        # fails: missing marker for node 'v'
        with pytest.raises(KeyError):
            graph._to_edge_type("A", "B", {"A": "-"})

    def test_to_adjacency(self):
        """Test the `_CoreGraph.to_adjacency` method across encodings."""
        # edge_type (default): each cell is the edge type oriented row -> column
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>"), ("C", "D", "--"), ("D", "E", "o>")])
        adj = graph.to_adjacency()  # default encoding="edge_type"
        assert adj.loc["A", "B"] == "->"
        assert adj.loc["B", "A"] == "<-"
        assert adj.loc["B", "C"] == "<>"
        assert adj.loc["C", "B"] == "<>"
        assert adj.loc["C", "D"] == "--"
        assert adj.loc["D", "E"] == "o>"
        assert adj.loc["E", "D"] == "<o"
        assert adj.loc["A", "C"] == 0  # no edge
        assert adj.loc["A", "A"] == 0  # diagonal

        # edge_type: coincident directed + bidirected (ADMG) -> aligned tuple of edge types
        # (zipping the two cells reconstructs each edge from both orientations)
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        adj = graph.to_adjacency()
        assert set(zip(adj.loc["A", "B"], adj.loc["B", "A"])) == {("->", "<-"), ("<>", "<>")}

        # binary: only supported for DAG/PDAG-style classes (directed/undirected edges); a class that
        # also supports bidirected/circle edges (here base _CoreGraph) is rejected on SUPPORTED_EDGE_TYPES.
        with pytest.raises(ValueError, match="DAG and PDAG"):
            _CoreGraph(edge_list=[("A", "B", "->")]).to_adjacency(encoding="binary")

        class _DirectedGraph(_CoreGraph):
            SUPPORTED_EDGE_TYPES = frozenset({"->", "<-", "--"})

        # binary: directed -> asymmetric 0/1; undirected -> symmetric
        graph = _DirectedGraph(edge_list=[("A", "B", "->"), ("A", "C", "->"), ("B", "C", "->"), ("D", "E", "--")])
        adj = graph.to_adjacency(encoding="binary")
        assert adj.loc["A", "B"] == 1 and adj.loc["B", "A"] == 0  # arc A->B (asymmetric)
        assert adj.loc["A", "C"] == 1 and adj.loc["B", "C"] == 1
        assert adj.loc["D", "E"] == 1 and adj.loc["E", "D"] == 1  # undirected (symmetric)

        # bnlearn is accepted as an alias of binary
        assert graph.to_adjacency(encoding="bnlearn").equals(graph.to_adjacency(encoding="binary"))

        # binary: coincident edges (e.g. a directed and an undirected edge between a pair) -> raise
        coincident = _DirectedGraph()
        coincident.add_edge("A", "B", "->")
        coincident.add_edge("A", "B", "--")
        with pytest.raises(ValueError, match="coincident"):
            coincident.to_adjacency(encoding="binary")

        # causal-learn: integer codes, mark at the ROW endpoint
        graph = _CoreGraph(edge_list=[("A", "B", "->")])
        adj = graph.to_adjacency(encoding="causal-learn")
        assert adj.loc["A", "B"] == -1 and adj.loc["B", "A"] == 1

        # causal-learn: coincident -> composite codes 4 / 5
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        adj = graph.to_adjacency(encoding="causal-learn")
        assert adj.loc["A", "B"] == 4 and adj.loc["B", "A"] == 5

        # pcalg amat.pag: column-indexed (1=circle, 2=arrowhead, 3=tail)
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("C", "D", "o>")])
        adj = graph.to_adjacency(encoding="pcalg")
        assert adj.loc["A", "B"] == 2 and adj.loc["B", "A"] == 3  # arrow at B, tail at A
        assert adj.loc["C", "D"] == 2 and adj.loc["D", "C"] == 1  # arrow at D, circle at C

        # pcalg: coincident edges cannot be represented -> raise
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        with pytest.raises(ValueError, match="pcalg"):
            graph.to_adjacency(encoding="pcalg")

        # fails: unknown encoding
        with pytest.raises(ValueError, match="encoding"):
            _CoreGraph(edge_list=[("A", "B", "->")]).to_adjacency(encoding="nonsense")

    def test_from_adjacency(self):
        """`from_adjacency` is the inverse of `to_adjacency` for every encoding; the constructed
        graph is of the class it is called on, with that class's edge-type restrictions."""
        import pandas as pd

        # edge_type round-trip, including parallel edges, circle marks, and an isolated node
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("A", "B", "<>"), ("B", "C", "o>"), ("C", "D", "--")])
        graph.add_node("E")
        assert _CoreGraph.from_adjacency(graph.to_adjacency()) == graph

        # binary round-trip (directed + undirected), class-typed result
        pdag = PDAG(edge_list=[("A", "B", "->"), ("B", "C", "--")])
        result = PDAG.from_adjacency(pdag.to_adjacency(encoding="binary"), encoding="binary")
        assert type(result) is PDAG
        assert result == pdag

        # pcalg round-trip; causal-learn round-trip including coincident (4/5-coded) edges
        mag = MAG(edge_list=[("X", "Y", "->"), ("Y", "Z", "<>")])
        assert MAG.from_adjacency(mag.to_adjacency(encoding="pcalg"), encoding="pcalg") == mag
        admg = ADMG(edge_list=[("X", "Y", "->"), ("X", "Y", "<>"), ("Y", "Z", "->")])
        assert ADMG.from_adjacency(admg.to_adjacency(encoding="causal-learn"), encoding="causal-learn") == admg

        # subclass edge-type restrictions are enforced: a PDAG cannot hold "<>"
        with pytest.raises(ValueError):
            PDAG.from_adjacency(ADMG(edge_list=[("A", "B", "<>")]).to_adjacency())

        # fails: unknown encoding / binary on a class with non-DAG/PDAG edge types /
        # mismatched index vs columns / mirror-inconsistent cells
        with pytest.raises(ValueError, match="encoding"):
            _CoreGraph.from_adjacency(graph.to_adjacency(), encoding="nonsense")
        with pytest.raises(ValueError, match="binary"):
            _CoreGraph.from_adjacency(pdag.to_adjacency(encoding="binary"), encoding="binary")
        bad_labels = pd.DataFrame(0, index=["A", "B"], columns=["B", "A"])
        with pytest.raises(ValueError, match="index"):
            _CoreGraph.from_adjacency(bad_labels)
        inconsistent = pd.DataFrame(0, index=["A", "B"], columns=["A", "B"], dtype=object)
        inconsistent.at["A", "B"] = "->"  # mirror cell left as 0
        with pytest.raises(ValueError, match="[Ii]nconsistent"):
            _CoreGraph.from_adjacency(inconsistent)

    def test_get_subgraph(self):
        """Test the `_CoreGraph.get_subgraph` method (class-preserving, node-induced, independent copy)."""
        # node-induced subgraph keeps only edges among the requested nodes, with their edge types preserved
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>"), ("C", "D", "--"), ("D", "E", "o>")])
        sub = graph.get_subgraph(["A", "B", "C"])
        assert type(sub) is _CoreGraph
        assert set(sub.nodes()) == {"A", "B", "C"}
        assert set(sub.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "<>")}

        # the result is an independent copy, not a networkx view: mutating one does not affect the other
        sub.add_edge("A", "C", "->")
        assert not graph.has_edge("A", "C")
        graph.remove_edge("A", "B", "->")
        assert sub.has_edge("A", "B", "->")

        # roles (exposures/outcomes/latents/custom) are carried over, restricted to the retained nodes
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "->")]
        graph = _CoreGraph(edge_list=edges, exposures=["A"], outcomes=["C"], latents=["D"])
        sub = graph.get_subgraph(["A", "B", "C"])
        check_graph_status(sub, 3, 2, {"A"}, {"C"}, set(), {"exposures": ["A"], "outcomes": ["C"]})

        # roles are deep-copied: adding a role to a node that already has one does not leak to the original
        sub.with_role("outcomes", ["A"], inplace=True)
        assert sub.outcomes == {"A", "C"} and graph.outcomes == {"C"}

        # a single-node list yields a one-node subgraph with no edges
        sub = graph.get_subgraph(["A"])
        assert set(sub.nodes()) == {"A"}
        assert sub.get_edges(data=True) == []

        # coincident edges (a directed and a bidirected edge between the same pair) are both preserved
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        graph.add_edge("B", "C", "->")
        sub = graph.get_subgraph(["A", "B"])
        assert sub.get_edge_type("A", "B") == {"->", "<>"}
        assert not sub.has_edge("B", "C")

        # the subclass is preserved (here ADMG)
        admg = ADMG(edge_list=[("A", "B", "->"), ("B", "C", "<>"), ("C", "D", "->")])
        sub = admg.get_subgraph(["B", "C", "D"])
        assert type(sub) is ADMG
        assert set(sub.get_edges(data=True)) == {("B", "C", "<>"), ("C", "D", "->")}

        # requesting a node that is not in the graph raises
        with pytest.raises(ValueError, match="not in graph"):
            admg.get_subgraph(["B", "Z"])

        # edge_types filter: keep only edges of the given (canonical) types; nodes default to all
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>"), ("C", "D", "--"), ("A", "D", "<-")])
        directed = graph.get_subgraph(edge_types={"->"})
        assert set(directed.nodes()) == {"A", "B", "C", "D"}  # all nodes retained
        # "->" matches both "->" and its reverse "<-" (same canonical type); get_edges returns canonical form
        assert set(directed.get_edges(data=True)) == {("A", "B", "->"), ("D", "A", "->")}

        # node and edge-type filters combine
        both = graph.get_subgraph(nodes=["A", "B", "C"], edge_types={"->", "<>"})
        assert set(both.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "<>")}

        # no arguments returns a full independent copy (like copy())
        assert graph.get_subgraph() == graph

        # get_ancestral_graph (which is built on get_subgraph) still returns the correct independent graph
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "--")])
        ancestral = graph.get_ancestral_graph(["C"])
        assert type(ancestral) is _CoreGraph
        assert set(ancestral.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}
        ancestral.add_edge("A", "C", "->")
        assert not graph.has_edge("A", "C")

    @pytest.mark.skipif(
        not _check_soft_dependencies("daft-pgm", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_to_daft(self):
        """Test `_CoreGraph.to_daft`: directed/undirected edges render; unsupported marks raise."""
        import daft

        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("C", "B", "<-"), ("C", "D", "--")], latents=["D"])
        # default layout returns a daft PGM object
        assert isinstance(graph.to_daft(), daft.PGM)
        # explicit dict positions, non-latex labels, and per-node / per-edge params are all accepted
        positions = {node: (i, 0) for i, node in enumerate(sorted(graph.nodes()))}
        assert isinstance(
            graph.to_daft(
                node_pos=positions,
                latex=False,
                node_params={"A": {"shape": "rectangle"}},
                edge_params={("A", "B"): {"label": "x"}},
            ),
            daft.PGM,
        )

        # invalid node_pos handling
        with pytest.raises(ValueError, match="node_pos"):
            graph.to_daft(node_pos="nonsense")
        with pytest.raises(ValueError, match="No position specified"):
            graph.to_daft(node_pos={"A": (0, 0)})

        # daft cannot represent bidirected or circle endpoints -> raise a clear error naming the edge type
        with pytest.raises(ValueError, match="daft"):
            ADMG(edge_list=[("A", "B", "<>")]).to_daft()
        with pytest.raises(ValueError, match="daft"):
            _CoreGraph(edge_list=[("A", "B", "o>")]).to_daft()

    @pytest.mark.skipif(
        not _check_soft_dependencies("pygraphviz", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_to_graphviz(self):
        """Test `_CoreGraph.to_graphviz`: every edge type maps to graphviz arrowtail/arrowhead."""
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>"), ("C", "D", "--"), ("D", "E", "o>")])
        agraph = graph.to_graphviz()
        assert agraph.is_directed()
        # mark -> arrow: '-' -> none, '>' -> normal, 'o' -> odot; mark at u is arrowtail, mark at v is arrowhead
        expected = {
            ("A", "B"): ("none", "normal"),  # ->
            ("B", "C"): ("normal", "normal"),  # <>
            ("C", "D"): ("none", "none"),  # --
            ("D", "E"): ("odot", "normal"),  # o>
        }
        for (u, v), (tail, head) in expected.items():
            edge = agraph.get_edge(u, v)
            assert edge.attr["dir"] == "both"
            assert edge.attr["arrowtail"] == tail
            assert edge.attr["arrowhead"] == head

    def test_get_topological_order(self):
        """Test `_CoreGraph.get_topological_order`: orders the directed part; raises only on circle marks."""
        # fully directed (including a `<-` edge, which is directed) -> a valid topological order
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("A", "C", "->"), ("B", "C", "->"), ("C", "D", "<-")])
        order = graph.get_topological_order()
        assert set(order) == set(graph.nodes())
        position = {node: i for i, node in enumerate(order)}
        for u, v, edge_type in graph.get_edges(data=True):
            source, target = (u, v) if edge_type == "->" else (v, u)
            assert position[source] < position[target]

        # isolated nodes are included
        graph.add_node("Z")
        assert "Z" in graph.get_topological_order()

        # bidirected (ADMG) and undirected (PDAG / chain graph) edges impose no ordering constraint:
        # they are ignored and the directed part is still ordered correctly.
        mixed = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>"), ("C", "D", "--"), ("D", "E", "->")])
        order = mixed.get_topological_order()
        position = {node: i for i, node in enumerate(order)}
        assert set(order) == set(mixed.nodes())
        assert position["A"] < position["B"] and position["D"] < position["E"]

        # circle endpoints (PAGs) leave ancestrality undetermined -> raise, naming the offending edge
        for edge_type in ("-o", "o-", "o>", "<o", "oo"):
            with pytest.raises(ValueError, match="circle"):
                _CoreGraph(edge_list=[("A", "B", "->"), ("X", "Y", edge_type)]).get_topological_order()

    def test_get_directed_graph(self):
        """Test `_CoreGraph.get_directed_graph`: an nx.DiGraph of the "->" edges only (canonical)."""
        import networkx as nx

        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("C", "D", "--"), ("E", "F", "<>"), ("G", "H", "<-")])
        graph.add_node("I")  # isolated nodes are preserved
        dg = graph.get_directed_graph()
        assert type(dg) is nx.DiGraph
        assert set(dg.nodes()) == {"A", "B", "C", "D", "E", "F", "G", "H", "I"}
        # only "->" edges, taken canonically ("<-" -> H->G); "--" and "<>" are dropped
        assert set(dg.edges()) == {("A", "B"), ("H", "G")}

    def test_get_skeleton(self):
        """Test `_CoreGraph.get_skeleton`: an undirected nx.Graph of the adjacency, edge types discarded."""
        import networkx as nx

        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>"), ("C", "D", "--"), ("D", "E", "o>")])
        graph.add_node("Z")  # isolated nodes are preserved
        skeleton = graph.get_skeleton()
        assert type(skeleton) is nx.Graph and not skeleton.is_directed()
        assert set(skeleton.nodes()) == {"A", "B", "C", "D", "E", "Z"}
        # every edge becomes an undirected adjacency, regardless of its original type
        assert {frozenset(edge) for edge in skeleton.edges()} == {
            frozenset({"A", "B"}),
            frozenset({"B", "C"}),
            frozenset({"C", "D"}),
            frozenset({"D", "E"}),
        }

        # coincident edges collapse to a single undirected edge, and the result is independent
        multi = _CoreGraph()
        multi.add_edge("A", "B", "->")
        multi.add_edge("A", "B", "<>")
        skeleton = multi.get_skeleton()
        assert skeleton.number_of_edges() == 1
        multi.remove_edge("A", "B", "->")
        assert skeleton.number_of_edges() == 1  # unaffected by later mutation of the original

    def test_do(self):
        """Test `_CoreGraph.do`: removes every edge with an arrowhead at the do-variable (its causes)."""
        # directed AND bidirected edges into A are removed; outgoing/undirected edges kept; original intact
        graph = _CoreGraph(edge_list=[("X", "A", "->"), ("Z", "A", "<>"), ("A", "Y", "->"), ("A", "B", "--")])
        result = graph.do("A")
        assert type(result) is _CoreGraph
        # compared via `==` (adjacency-based) so symmetric-edge node order doesn't matter; X/Z are kept isolated
        expected = _CoreGraph(edge_list=[("A", "Y", "->"), ("A", "B", "--")])
        expected.add_nodes_from(["X", "Z"])
        assert result == expected
        # not inplace: the original is unchanged
        assert graph == _CoreGraph(edge_list=[("X", "A", "->"), ("Z", "A", "<>"), ("A", "Y", "->"), ("A", "B", "--")])

        # an arrowhead at A is incoming and removed regardless of the other endpoint (here Z o> A)
        assert _CoreGraph(edge_list=[("Z", "A", "o>")]).do("A").get_edges(data=True) == []
        # a circle endpoint AT the do-variable is non-invariant (could be tail or arrowhead) -> raise
        with pytest.raises(ValueError, match="circle"):
            _CoreGraph(edge_list=[("A", "B", "o>")]).do("A")  # circle at A
        with pytest.raises(ValueError, match="circle"):
            _CoreGraph(edge_list=[("A", "B", "oo")]).do("A")  # circle at A

        # multiple do-variables remove all edges with an arrowhead at A or B (incl. A->B and Y<>B)
        graph = _CoreGraph(edge_list=[("X", "A", "->"), ("A", "B", "->"), ("Y", "B", "<>")])
        result = graph.do(["A", "B"])
        assert result.get_edges(data=True) == []
        assert set(result.nodes()) == {"X", "A", "B", "Y"}

        # works on a subclass with restricted edge types (ADMG has no circle/`<o` types to query)
        admg = ADMG(edge_list=[("X", "A", "->"), ("Z", "A", "<>"), ("A", "Y", "->")])
        result = admg.do("A")
        assert type(result) is ADMG
        assert set(result.get_edges(data=True)) == {("A", "Y", "->")}

        # inplace=True returns and mutates the same graph
        graph = _CoreGraph(edge_list=[("X", "A", "->"), ("A", "Y", "->")])
        same = graph.do("A", inplace=True)
        assert same is graph
        assert set(graph.get_edges(data=True)) == {("A", "Y", "->")}

        # node not in the graph raises
        with pytest.raises(ValueError, match="not found"):
            _CoreGraph(edge_list=[("X", "A", "->")]).do("Q")

        # a tuple is a single (hashable) node, not a collection -- same convention as `get_parents`
        graph = _CoreGraph(edge_list=[("X", ("A", 0), "->"), (("A", 0), "Y", "->")])
        result = graph.do(("A", 0))
        assert set(result.get_edges(data=True)) == {(("A", 0), "Y", "->")}

    def test_get_roots_and_leaves(self):
        """Test `get_roots`/`get_leaves` (mark-based): root = all tails at node, leaf = all arrowheads at node."""
        g = _CoreGraph(edge_list=[("X", "A", "->"), ("A", "Y", "->"), ("A", "B", "--"), ("Z", "Y", "<>")])
        g.add_node("Iso")  # an isolated node is both a root and a leaf
        # roots = only tails at the node: X (-> out), B (undirected), Iso; A has an incoming arrowhead, Y/Z don't
        assert g.get_roots() == {"X", "B", "Iso"}
        # leaves = only arrowheads at the node: Y (incoming -> and <>), Z (bidirected), Iso; X/A/B have tails
        assert g.get_leaves() == {"Y", "Z", "Iso"}

        # a circle endpoint at a node makes it neither a root nor a leaf
        c = _CoreGraph(edge_list=[("P", "Q", "o>"), ("Q", "R", "->")])
        assert "P" not in c.get_roots() and "P" not in c.get_leaves()

        # subclass with restricted edge types (ADMG has no circle/`--`/`<o` types to query)
        admg = ADMG(edge_list=[("X", "A", "->"), ("Z", "A", "<>")])
        assert admg.get_roots() == {"X"}
        assert admg.get_leaves() == {"A", "Z"}

    def test_has_path(self):
        """Test `has_path`: reachability from source to target following only `edge_types` edges."""
        g = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--"), ("E", "B", "<>")])
        assert g.has_path("A", "C", edge_types="->") is True
        assert g.has_path("A", "D", edge_types="->") is False  # C--D is not directed
        assert g.has_path("A", "D", edge_types={"->", "--"}) is True
        assert g.has_path("A", "E") is True  # None follows any edge type (E<>B)
        assert g.has_path("A", "E", edge_types="->") is False
        assert g.has_path("A", "A", edge_types="->") is True  # reflexive
        with pytest.raises(ValueError):
            g.has_path("A", "Z")

    def test_get_all_paths(self):
        """Test `get_all_paths`: all simple paths from source to target following only `edge_types` edges."""
        g = _CoreGraph(
            edge_list=[("A", "B", "->"), ("A", "C", "->"), ("B", "D", "->"), ("C", "D", "->"), ("A", "D", "->")]
        )
        paths = g.get_all_paths("A", "D", edge_types="->")
        assert {tuple(p) for p in paths} == {("A", "B", "D"), ("A", "C", "D"), ("A", "D")}
        assert g.get_all_paths("A", "A", edge_types="->") == [["A"]]  # trivial path
        assert g.get_all_paths("B", "C", edge_types="->") == []  # no directed B -> C
        with pytest.raises(ValueError):
            g.get_all_paths("A", "Z")
