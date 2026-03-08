#!/usr/bin/env python3

import pytest

from pgmpy.base import ADMG, DAG, PDAG
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
    return _CoreGraph(ebunch=edges)


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
    return _CoreGraph(ebunch=edges)


def sample_graph3():
    """
    sample graph for testing node searching(`get_*`) method of `_CoreGraph` class.

    Notes
    -----
    Used `base_graph` from test_AncestralBase.py.
    Expected to be same as tests in test_AncestralBase.py.
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
    return _CoreGraph(ebunch=edges)


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


class TestCoreGraph:
    def test_init_empty(self):
        """Test the initialization of an empty `_CoreGraph`"""
        graph = _CoreGraph()
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_init_with_nodes(self):
        """Test the initialization of a `_CoreGraph` with nodes."""
        edges = [("A", "B", "->"), ("B", "C", "->")]
        graph = _CoreGraph(ebunch=edges)

        assert sorted(graph.nodes) == ["A", "B", "C"]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_init_with_edges(self):
        """Test the initialization of a `_CoreGraph` with edges."""
        edges = [("A", "B", "--"), ("A", "B", "-o"), ("B", "C", "<>")]
        graph = _CoreGraph(ebunch=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": "-"}),
            ("A", "B", 1, {"A": "-", "B": "o"}),
            ("B", "C", 0, {"B": ">", "C": ">"}),
        ]
        check_graph_status(graph, 3, 3, set(), set(), set(), {})

    def test_init_with_exposures(self):
        """Test the initialization of a `_CoreGraph` with exposures."""
        edges = [("A", "B", "->")]
        graph = _CoreGraph(ebunch=edges, exposures=["A"])

        assert sorted(graph.exposures) == ["A"]
        check_graph_status(graph, 2, 1, {"A"}, set(), set(), {"exposures": ["A"]})

    def test_init_with_outcomes(self):
        """Test the initialization of a `_CoreGraph` with outcomes."""
        edges = [("A", "B", "->")]
        graph = _CoreGraph(ebunch=edges, outcomes=["B"])

        assert sorted(graph.outcomes) == ["B"]
        check_graph_status(graph, 2, 1, set(), {"B"}, set(), {"outcomes": ["B"]})

    def test_init_with_latents(self):
        """Test the initialization of a `_CoreGraph` with latents."""
        edges = [("A", "B", "->")]
        graph = _CoreGraph(ebunch=edges, latents=["A"])

        assert sorted(graph.latents) == ["A"]
        check_graph_status(graph, 2, 1, set(), set(), {"A"}, {"latents": ["A"]})

    def test_init_with_roles(self):
        """Test the initialization of a `_CoreGraph` with roles."""
        edges = [("A", "B", "->")]
        graph = _CoreGraph(ebunch=edges, roles={"test_role": ["A"]})

        assert sorted(graph.get_roles()) == ["test_role"]
        check_graph_status(graph, 2, 1, set(), set(), set(), {"test_role": ["A"]})

    def test_init_with_all_values(self):
        """Test the initialization of a `_CoreGraph` with all values."""
        edges = [("A", "B", "->"), ("B", "C", "oo"), ("C", "D", "--")]
        graph = _CoreGraph(
            ebunch=edges,
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

    def test_init_fails(self):
        """Test failing the initialization of a `_CoreGraph`."""
        # Task8: Test failing the initialization of a `_CoreGraph` with values.
        graph = _CoreGraph()

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), (None, "A", "->"), ("B", "C", "->")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), ("A", None, "->"), ("B", "C", "->")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # same node error
            edges = [("A", "A", "->")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # same nodes error
            edges = [("A", "B", "->"), ("A", "A", "->"), ("C", "D", "--")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `edge_type` value
            edges = [("A", "B", "-->")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `edge_type` values
            edges = [("A", "B", "->"), ("A", "C", "o-->"), ("C", "D", "--")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # Granting a role to a node that is not owned.
            roles = {"test_role": "A"}
            graph = _CoreGraph(roles=roles)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # Granting a role to a node that is not owned.
            edges = [("A", "B", "->")]
            roles = {"test_role1": "A", "test_role2": "C", "test_role3": "B"}
            graph = _CoreGraph(ebunch=edges, roles=roles)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_add_directed_edge(self):
        """Test adding the direct edge of a `_CoreGraph`."""
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

    def test_add_undirected_edge(self):
        """Test adding the undirect edge of a `_CoreGraph`."""
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

    def test_add_bidirected_edge(self):
        """Test adding the bidirect edge of a `_CoreGraph`."""
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

    def test_add_unknown_edge(self):
        """Test adding the unknown edge of a `_CoreGraph`."""
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

    def test_add_multiedges(self):
        """Test adding multiedges of a `_CoreGraph`."""
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

    def test_add_edge_with_kwargs(self):
        """Test adding edge of with kwargs."""
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->", weight=5)
        graph.add_edge("B", "C", "->", weight=8)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"weight": 5, "A": "-", "B": ">"}),
            ("B", "C", 0, {"weight": 8, "B": "-", "C": ">"}),
        ]

    def test_add_edge_fails(self):
        """Test failing add edge of a `_CoreGraph`."""
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.add_edge("A", "A", "->")

        with pytest.raises(ValueError):
            graph.add_edge("A", "B", "-->")

        with pytest.raises(ValueError):
            graph.add_edge("A", "B", "Invalid_value")

        assert not graph.has_edge("A", "B")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_add_directed_edges_from(self):
        """Test adding the direct edges of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("B", "C", "->")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": ">"}),
            ("B", "C", 0, {"B": "-", "C": ">"}),
        ]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_add_undirected_edges_from(self):
        """Test adding the undirect edges of a `_CoreGraph`."""
        edges = [("A", "B", "--"), ("B", "C", "--")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": "-"}),
            ("B", "C", 0, {"B": "-", "C": "-"}),
        ]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_add_bidirected_edges_from(self):
        """Test adding the bidirect edges of a `_CoreGraph`."""
        edges = [("A", "B", "<>"), ("B", "C", "<>")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": ">", "B": ">"}),
            ("B", "C", 0, {"B": ">", "C": ">"}),
        ]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_add_unknown_edges_from(self):
        """Test adding the unknown edges of a `_CoreGraph`."""
        edges = [("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": "o"}),
            ("B", "C", 0, {"B": "o", "C": "-"}),
            ("C", "D", 0, {"C": "o", "D": "o"}),
        ]
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_add_various_edges_from(self):
        """Test adding the various edge of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("B", "C", "--"), ("C", "D", "<>")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": ">"}),
            ("B", "C", 0, {"B": "-", "C": "-"}),
            ("C", "D", 0, {"C": ">", "D": ">"}),
        ]
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_add_multiedges_from(self):
        """Test adding multiedges of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("A", "B", "--"), ("A", "B", "oo")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": ">"}),
            ("A", "B", 1, {"A": "-", "B": "-"}),
            ("A", "B", 2, {"A": "o", "B": "o"}),
        ]
        check_graph_status(graph, 2, 3, set(), set(), set(), {})

    def test_add_edges_from_fails(self):
        """Test failing add edges of a `_CoreGraph`."""
        graph = _CoreGraph()

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), (None, "A", "->"), ("B", "C", "->")]
            graph.add_edges_from(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), ("A", None, "->"), ("B", "C", "->")]
            graph.add_edges_from(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # miss `edge_type` value
            edges = [("A", "B", "->"), ("A", "C"), ("B", "C", "->")]
            graph.add_edges_from(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # same node error
            edges = [("A", "B", "->"), ("A", "A", "->"), ("B", "C", "->")]
            graph.add_edges_from(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `edge_type` value
            edges = [("A", "B", "->"), ("B", "C", "-->"), ("C", "D", "->")]
            graph.add_edges_from(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_remove_directed_edge(self):
        """Test removing the direct edge of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("B", "C", "<-")]
        graph = _CoreGraph(ebunch=edges)

        graph.remove_edge("A", "B", "->")
        graph.remove_edge("B", "C", "<-")

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

    def test_remove_undirected_edge(self):
        """Test removing the undirect edge of a `_CoreGraph`."""
        edges = [("A", "B", "--"), ("B", "C", "--")]
        graph = _CoreGraph(ebunch=edges)

        graph.remove_edge("A", "B", "--")
        graph.remove_edge("B", "C", "--")

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

    def test_remove_bidirected_edge(self):
        """Test removing the bidirect edge of a `_CoreGraph`."""
        edges = [("A", "B", "<>"), ("B", "C", "<>")]
        graph = _CoreGraph(ebunch=edges)

        graph.remove_edge("A", "B", "<>")
        graph.remove_edge("B", "C", "<>")

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

    def test_remove_unknown_edge(self):
        """Test removing the unknown edge of a `_CoreGraph`."""
        edges = [("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")]
        graph = _CoreGraph(ebunch=edges)

        graph.remove_edge("A", "B", "-o")
        graph.remove_edge("B", "C", "o-")
        graph.remove_edge("C", "D", "oo")

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")
        assert not graph.has_edge("C", "D")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 4, 0, set(), set(), set(), {})

    def test_remove_multiedges(self):
        """Test removing multiedges of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("A", "B", "->"), ("A", "B", "--")]
        graph = _CoreGraph(ebunch=edges)

        graph.remove_edge("A", "B", "->")
        graph.remove_edge("A", "B", "--")

        assert not graph.has_edge("A", "B")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 2, 0, set(), set(), set(), {})

    def test_remove_no_edge_type(self):
        """Test removing edges of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("A", "B", "->"), ("A", "B", "--")]
        graph = _CoreGraph(ebunch=edges)
        graph.remove_edge("A", "B")
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

    def test_remove_edge_fails(self):
        """Test failing remove edge of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("B", "C", "->")]

        graph = _CoreGraph(ebunch=edges)
        graph.remove_edge("A", "B", "->")
        with pytest.raises(ValueError):  # invalid `u`, `v` value
            graph.remove_edge(None, "C", "->")
        check_graph_status(graph, 3, 1, set(), set(), set(), {})

        graph = _CoreGraph(ebunch=edges)
        graph.remove_edge("A", "B", "->")
        with pytest.raises(ValueError):  # invalid `u`, `v` value
            graph.remove_edge("B", None, "->")
        check_graph_status(graph, 3, 1, set(), set(), set(), {})

        graph = _CoreGraph(ebunch=edges)
        graph.remove_edge("A", "B", "->")
        with pytest.raises(ValueError):  # same node error
            graph.remove_edge("B", "B", "->")
        check_graph_status(graph, 3, 1, set(), set(), set(), {})

        graph = _CoreGraph(ebunch=edges)
        graph.remove_edge("A", "B", "->")
        with pytest.raises(ValueError):  # invalid `edge_type` value
            graph.remove_edge("B", "C", "invalid_value")
        check_graph_status(graph, 3, 1, set(), set(), set(), {})

    def test_remove_directed_edges_from(self):
        """Test removing the direct edges of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("B", "C", "<-")]
        graph = _CoreGraph(ebunch=edges)

        graph.remove_edges_from(ebunch=edges)

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

    def test_remove_undirected_edges_from(self):
        """Test removing the undirect edges of a `_CoreGraph`."""
        edges = [("A", "B", "--"), ("B", "C", "--")]
        graph = _CoreGraph(ebunch=edges)

        graph.remove_edges_from(ebunch=edges)

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

    def test_remove_bidirected_edges_from(self):
        """Test removing the bidirect edges of a `_CoreGraph`."""
        edges = [("A", "B", "<>"), ("B", "C", "<>")]
        graph = _CoreGraph(ebunch=edges)

        graph.remove_edges_from(ebunch=edges)

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

    def test_remove_unknown_edges_from(self):
        """Test removing the unknown edges of a `_CoreGraph`."""
        edges = [("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")]
        graph = _CoreGraph(ebunch=edges)

        graph.remove_edges_from(ebunch=edges)

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")
        assert not graph.has_edge("C", "D")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 4, 0, set(), set(), set(), {})

    def test_remove_various_edges_from(self):
        """Test removing the various edge of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("B", "C", "--"), ("C", "D", "<o")]
        graph = _CoreGraph(ebunch=edges)

        graph.remove_edges_from(ebunch=edges)

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")
        assert not graph.has_edge("C", "D")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 4, 0, set(), set(), set(), {})

    def test_remove_multiedges_from(self):
        """Test removing multiedges of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("A", "B", "->"), ("A", "B", "--")]
        graph = _CoreGraph(ebunch=edges)

        del_edges = [("A", "B", "->"), ("A", "B", "--")]
        graph.remove_edges_from(ebunch=del_edges)

        assert not graph.has_edge("A", "B")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 2, 0, set(), set(), set(), {})

    def test_remove_edges_from_fails(self):
        """Test failing remove edges of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("B", "C", "->")]

        graph = _CoreGraph(ebunch=edges)
        with pytest.raises(ValueError):  # invalid `u`, `v` value
            graph.remove_edges_from([("A", "B", "->"), (None, "C", "->")])
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        graph = _CoreGraph(ebunch=edges)
        with pytest.raises(ValueError):  # invalid `u`, `v` value
            graph.remove_edges_from([("A", "B", "->"), ("B", None, "->")])
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        graph = _CoreGraph(ebunch=edges)
        with pytest.raises(ValueError):  # miss `edge_type` value
            graph.remove_edges_from([("A", "B", "->"), ("B", "C")])
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        graph = _CoreGraph(ebunch=edges)
        with pytest.raises(ValueError):  # same node error
            graph.remove_edges_from([("A", "B", "->"), ("B", "B", "->")])
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        graph = _CoreGraph(ebunch=edges)
        with pytest.raises(ValueError):  # invalid `edge_type` value
            graph.remove_edges_from([("A", "B", "->"), ("B", "C", "invalid_value")])
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_equality_empty(self):
        """Test the `__eq__` method of the empty `_CoreGraph` class."""
        graph = _CoreGraph()
        other = _CoreGraph()

        assert graph.__eq__(other) == True
        assert other.__eq__(graph) == True

        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_equality_with_values(self):
        """Test the `__eq__` method of the `_CoreGraph` class with values."""
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]
        roles = {"test_role": ["B"]}
        graph = _CoreGraph(
            ebunch=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

        other = _CoreGraph()
        other.add_edges_from(ebunch=edges)
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

    def test_equality_different_graphs(self):
        """Test the `__eq__` method with different graphs."""
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]
        roles = {"test_role": ["B"]}
        graph = _CoreGraph(
            ebunch=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

        # Different class
        other_dag = DAG()
        other_admg = ADMG()
        other_pdag = PDAG()

        assert graph.__eq__(other_dag) == False
        assert other_dag.__eq__(graph) == False
        assert graph.__eq__(other_admg) == False
        assert other_admg.__eq__(graph) == False
        assert graph.__eq__(other_pdag) == False
        assert other_pdag.__eq__(graph) == False

        # Different ebunch
        other_ebunch = _CoreGraph(
            ebunch=[
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
        assert graph.__eq__(other_ebunch) == False
        assert other_ebunch.__eq__(graph) == False

        # Different exposures
        other_exp = _CoreGraph(
            ebunch=edges,
            exposures=["B"],
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )
        assert graph.__eq__(other_exp) == False
        assert other_exp.__eq__(graph) == False

        # Different outcomes
        other_out = _CoreGraph(
            ebunch=edges,
            exposures=exposures,
            outcomes=["A"],
            latents=latents,
            roles=roles,
        )
        assert graph.__eq__(other_out) == False
        assert other_out.__eq__(graph) == False

        # Different latents
        other_lat = _CoreGraph(
            ebunch=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=["B"],
            roles=roles,
        )
        assert graph.__eq__(other_lat) == False
        assert other_lat.__eq__(graph) == False

        # Different roles
        other_roles = _CoreGraph(
            ebunch=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles={"new_role": ["A"]},
        )
        assert graph.__eq__(other_roles) == False
        assert other_roles.__eq__(graph) == False

        # Not a graph class
        other_str = "not a graph"
        assert graph.__eq__(other_str) == False

    def test_equality_fails(self):
        """Test failing the `__eq__` method of the `_CoreGraph` class."""
        graph = _CoreGraph()
        with pytest.raises(TypeError):
            graph.__eq__()
        with pytest.raises(TypeError):
            graph.__eq__(_CoreGraph(), _CoreGraph())

        with pytest.raises(AssertionError):
            # Not a graph class
            other_str = "not a graph"
            assert other_str.__eq__(graph) == False

    def test_copy_empty(self):
        """Test the `copy` method of the empty `_CoreGraph` class."""
        graph = _CoreGraph()
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_copy_with_nodes(self):
        """Test the `copy` method of a `_CoreGraph` with nodes."""
        nodes = ["A", "B", "C"]
        graph = _CoreGraph()
        graph.add_nodes_from(nodes)
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

    def test_copy_with_ebunch1(self):
        """Test the `copy` method of a `_CoreGraph` with an ebunch."""
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        graph = _CoreGraph(ebunch=edges)
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_copy_with_ebunch2(self):
        """Test the `copy` method of a `_CoreGraph` with an ebunch."""
        graph = _CoreGraph()
        graph.add_edge("A", "C", "->")
        graph.add_edge("C", "B", "<-")
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_copy_with_attributes(self):
        """Test the `copy` method of a `_CoreGraph` with attributes."""
        edges = [("A", "B", "->"), ("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]

        graph = _CoreGraph(
            ebunch=edges, exposures=exposures, outcomes=outcomes, latents=latents
        )
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        check_graph_status(
            graph,
            4,
            4,
            {"A"},
            {"C"},
            {"D"},
            {
                "exposures": ["A"],
                "outcomes": ["C"],
                "latents": ["D"],
            },
        )

    def test_copy_with_roles(self):
        """Test the `copy` method of a `_CoreGraph` with roles."""
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        roles = {"test_role": ["A", "B"]}
        graph = _CoreGraph(ebunch=edges, roles=roles)
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        check_graph_status(
            graph,
            4,
            3,
            set(),
            set(),
            set(),
            {
                "test_role": ["A", "B"],
            },
        )

    def test_copy_with_all_values(self):
        """Test the `copy` method of a `_CoreGraph` with all values."""
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]
        roles = {"test_role": ["B"]}
        graph = _CoreGraph(
            ebunch=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

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

    def test_copy_fails(self):
        """Test failing the `copy` method of the `_CoreGraph` class."""
        graph = _CoreGraph()
        with pytest.raises(TypeError):
            graph.copy("invalid_value")

    # get_neighbors
    def test_get_neighbors_with_direct_edges(self):
        """Test `get_neighbors` method of the `_CoreGraph` class with direct edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="->")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # Test2: reverse edge.
        graph = sample_graph1(edge_type="<-")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_neighbors_with_undirect_edges(self):
        """Test `get_neighbors` method of the `_CoreGraph` class with undirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="--")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_neighbors_with_bidirect_edges(self):
        """Test `get_neighbors` method of the `_CoreGraph` class with bidirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="<>")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_neighbors_with_unknown_edges(self):
        """Test `get_neighbors` method of the `_CoreGraph` class with unknown edges."""
        graph = sample_graph1(edge_type="o>")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_neighbors_with_multiedges(self):
        """Test `get_neighbors` method of the `_CoreGraph` class with multiedges."""
        graph = sample_graph3()

        all_nodes_except_A = {"B", "C", "D", "E", "F", "G", "H", "I", "J"}
        assert graph.get_neighbors("A") == all_nodes_except_A
        assert graph.get_neighbors("B") == {"A", "X"}
        assert graph.get_neighbors("C") == {"A", "Y"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

    def test_get_neighbors_with_edge_type_values(self):
        """Test `get_neighbors` method of the `_CoreGraph` class with edge_type values."""
        graph = sample_graph1(edge_type="->")
        assert graph.get_neighbors("C", "->") == {"D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_neighbors("B", "->") == {"C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph1(edge_type="->")
        assert graph.get_neighbors("C", "<-") == {"B"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_neighbors("B", "<-") == {"A"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_neighbors("C", "o-") == {"B"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_neighbors("B", "o-") == {"A"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph3()
        assert graph.get_neighbors("A", "->") == {"B"}
        assert graph.get_neighbors("A", "<-") == {"C"}
        assert graph.get_neighbors("A", "<>") == {"E"}
        assert graph.get_neighbors("B", "->") == {"X"}
        assert graph.get_neighbors("B", "<-") == {"A"}
        assert graph.get_neighbors("C", "->") == {"A"}
        assert graph.get_neighbors("C", "<-") == {"Y"}
        assert graph.get_neighbors("C", "<>") == set()
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

    def test_get_neighbors_fails(self):
        """Test failing `get_neighbors` method of the `_CoreGraph` class"""
        # Test1: The `_CoreGraph` do not have any nodes.
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_neighbors("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # Test2: The `_CoreGraph` have nodes. But, do not have any edges.
        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_neighbors("A") == set()
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        # Test3: Wrong input node values.
        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_neighbors()
        with pytest.raises(ValueError):
            graph.get_neighbors(1)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # Test4: Wrong input edge_type values.
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")

        with pytest.raises(ValueError):
            graph.get_neighbors("A", "wrong_edge")
        check_graph_status(graph, 2, 1, set(), set(), set(), {})

    # get_parents
    def test_get_parents_with_direct_edges(self):
        """Test `get_parents` method of the `_CoreGraph` class with direct edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="->")
        assert graph.get_parents("C") == {"B"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_parents("B") == {"A"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # Test2: reverse edge.
        graph = sample_graph1(edge_type="<-")
        assert graph.get_parents("C") == {"D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_parents("B") == {"C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_parents_with_undirect_edges(self):
        """Test `get_parents` method of the `_CoreGraph` class with undirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="--")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_parents_with_bidirect_edges(self):
        """Test `get_parents` method of the `_CoreGraph` class with bidirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="<>")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_parents_with_unknown_edges(self):
        """Test `get_parents` method of the `_CoreGraph` class with unknown edges."""
        graph = sample_graph1(edge_type="o>")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_parents_with_multiedges(self):
        """Test `get_parents` method of the `_CoreGraph` class with multiedges."""
        graph = sample_graph3()
        assert graph.get_parents("A") == {"C"}
        assert graph.get_parents("B") == {"A"}
        assert graph.get_parents("C") == {"Y"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

    def test_get_parents_fails(self):
        """Test failing `get_parents` method of the `_CoreGraph` class"""
        # Test1: The `_CoreGraph` do not have any nodes.
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_parents("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # Test2: The `_CoreGraph` have nodes. But, do not have any edges.
        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_parents("A") == set()
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        # Test3: Wrong input values.
        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_parents()
        with pytest.raises(ValueError):
            graph.get_parents(1)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    # get_spouses
    def test_get_spouses_with_direct_edges(self):
        """Test `get_spouses` method of the `_CoreGraph` class with direct edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="->")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # Test2: reverse edge.
        graph = sample_graph1(edge_type="<-")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_spouses_with_undirect_edges(self):
        """Test `get_spouses` method of the `_CoreGraph` class with undirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="--")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_spouses_with_bidirect_edges(self):
        """Test `get_spouses` method of the `_CoreGraph` class with bidirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="<>")
        assert graph.get_spouses("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_spouses("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_spouses_with_unknown_edges(self):
        """Test `get_spouses` method of the `_CoreGraph` class with unknown edges."""
        graph = sample_graph1(edge_type="o>")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_spouses_with_multiedges(self):
        """Test `get_spouses` method of the `_CoreGraph` class with multiedges."""
        graph = sample_graph3()
        assert graph.get_spouses("A") == {"E"}
        assert graph.get_spouses("E") == {"A"}
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

    def test_get_spouses_fails(self):
        """Test failing `get_spouses` method of the `_CoreGraph` class"""
        # Test1: The `_CoreGraph` do not have any nodes.
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_spouses("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # Test2: The `_CoreGraph` have nodes. But, do not have any edges.
        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_spouses("A") == set()
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        # Test3: Wrong input values.
        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_spouses()
        with pytest.raises(ValueError):
            graph.get_spouses(1)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    # get_children
    def test_get_children_with_direct_edges(self):
        """Test `get_children` method of the `_CoreGraph` class with direct edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="->")
        assert graph.get_children("C") == {"D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_children("B") == {"C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # Test2: reverse edge.
        graph = sample_graph1(edge_type="<-")
        assert graph.get_children("C") == {"B"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_children("B") == {"A"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_children_with_undirect_edges(self):
        """Test `get_children` method of the `_CoreGraph` class with undirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="--")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_children_with_bidirect_edges(self):
        """Test `get_children` method of the `_CoreGraph` class with bidirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="<>")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_children_with_unknown_edges(self):
        """Test `get_children` method of the `_CoreGraph` class with unknown edges."""
        graph = sample_graph1(edge_type="o>")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_children_with_multiedges(self):
        """Test `get_children` method of the `_CoreGraph` class with multiedges."""
        graph = sample_graph3()
        assert graph.get_children("A") == {"B"}
        assert graph.get_children("B") == {"X"}
        assert graph.get_children("Y") == {"C"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

    def test_get_children_fails(self):
        """Test failing `get_children` method of the `_CoreGraph` class"""
        # Test1: The `_CoreGraph` do not have any nodes.
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_children("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # Test2: The `_CoreGraph` have nodes. But, do not have any edges.
        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_children("A") == set()
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        # Test3: Wrong input values.
        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_children()
        with pytest.raises(ValueError):
            graph.get_children(1)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    # get_ancestors
    def test_get_ancestors_with_direct_edges(self):
        """Test `get_ancestors` method of the `_CoreGraph` class with direct edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="->")
        assert graph.get_ancestors("C") == {"A", "B", "C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_ancestors("B") == {"A", "B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # Test2: reverse edge.
        graph = sample_graph1(edge_type="<-")
        assert graph.get_ancestors("C") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_ancestors("B") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_ancestors_with_undirect_edges(self):
        """Test `get_ancestors` method of the `_CoreGraph` class with undirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="--")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_ancestors_with_bidirect_edges(self):
        """Test `get_ancestors` method of the `_CoreGraph` class with bidirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="<>")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_ancestors_with_unknown_edges(self):
        """Test `get_ancestors` method of the `_CoreGraph` class with unknown edges."""
        graph = sample_graph1(edge_type="o>")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_ancestors_with_multiedges(self):
        """Test `get_ancestors` method of the `_CoreGraph` class with multiedges."""
        graph = sample_graph3()
        assert graph.get_ancestors("A") == {"A", "C", "Y"}
        assert graph.get_ancestors("X") == {"X", "B", "A", "C", "Y"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

    def test_get_ancestors_fails(self):
        """Test failing `get_ancestors` method of the `_CoreGraph` class"""
        # Test1: The `_CoreGraph` do not have any nodes.
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_ancestors("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # Test2: The `_CoreGraph` have nodes. But, do not have any edges.
        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_ancestors("A") == {"A"}
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        # Test3: Wrong input values.
        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_ancestors()
        with pytest.raises(ValueError):
            graph.get_ancestors("1")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    # get_descendants
    def test_get_descendants_with_direct_edges(self):
        """Test `get_descendants` method of the `_CoreGraph` class with direct edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="->")
        assert graph.get_descendants("C") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_descendants("B") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # Test2: reverse edge.
        graph = sample_graph1(edge_type="<-")
        assert graph.get_descendants("C") == {"A", "B", "C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_descendants("B") == {"A", "B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_descendants_with_undirect_edges(self):
        """Test `get_descendants` method of the `_CoreGraph` class with undirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="--")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_descendants_with_bidirect_edges(self):
        """Test `get_descendants` method of the `_CoreGraph` class with bidirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="<>")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_descendants_with_unknown_edges(self):
        """Test `get_descendants` method of the `_CoreGraph` class with unknown edges."""
        graph = sample_graph1(edge_type="o>")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_descendants_with_multiedges(self):
        """Test `get_descendants` method of the `_CoreGraph` class with multiedges."""
        graph = sample_graph3()
        assert graph.get_descendants("A") == {"A", "B", "X"}
        assert graph.get_descendants("Y") == {"Y", "C", "A", "B", "X"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

    def test_get_descendants_fails(self):
        """Test failing `get_descendants` method of the `_CoreGraph` class"""
        # Test1: The `_CoreGraph` do not have any nodes.
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_descendants("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # Test2: The `_CoreGraph` have nodes. But, do not have any edges.
        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_descendants("A") == {"A"}
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        # Test3: Wrong input values.
        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_descendants()
        with pytest.raises(ValueError):
            graph.get_descendants("1")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    # get_reachable_nodes
    def test_get_reachable_nodes_with_direct_edges(self):
        """Test `get_reachable_nodes` method of the `_CoreGraph` class with direct edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="->")
        assert graph.get_reachable_nodes("C", "->") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_reachable_nodes("B", "->") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # Test2: reverse edge.
        graph = sample_graph1(edge_type="<-")
        assert graph.get_reachable_nodes("C", "<-") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_reachable_nodes("B", "<-") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_reachable_nodes_with_undirect_edges(self):
        """Test `get_reachable_nodes` method of the `_CoreGraph` class with undirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="--")
        assert graph.get_reachable_nodes("C", "--") == {"A", "B", "C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_reachable_nodes("B", "--") == {"A", "B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_reachable_nodes_with_bidirect_edges(self):
        """Test `get_reachable_nodes` method of the `_CoreGraph` class with bidirect edges."""
        # Test1: edge.
        graph = sample_graph1(edge_type="<>")
        assert graph.get_reachable_nodes("C", "<>") == {"A", "B", "C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_reachable_nodes("B", "<>") == {"A", "B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_reachable_nodes_with_unknown_edges(self):
        """Test `get_reachable_nodes` method of the `_CoreGraph` class with unknown edges."""
        graph = sample_graph1(edge_type="o>")
        assert graph.get_reachable_nodes("C", "o>") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_reachable_nodes("C", "<o") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_reachable_nodes("C", "-o") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_reachable_nodes("C", "o-") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_reachable_nodes("C", "oo") == {"A", "B", "C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_reachable_nodes("B", "o>") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_reachable_nodes("B", "<o") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_reachable_nodes("B", "-o") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_reachable_nodes("B", "o-") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_reachable_nodes("B", "oo") == {"A", "B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_get_reachable_nodes_with_multiedges(self):
        """Test `get_reachable_nodes` method of the `_CoreGraph` class with multiedges."""
        graph = sample_graph3()
        assert graph.get_reachable_nodes("A", "->") == {
            "A",
            "B",
            "X",
        }
        assert graph.get_reachable_nodes("A", "<-") == {
            "A",
            "C",
            "Y",
        }
        assert graph.get_reachable_nodes("A", "oo") == {"A", "D"}
        assert graph.get_reachable_nodes("A", "<>") == {"A", "E"}
        assert graph.get_reachable_nodes("A", "--") == {"A", "F"}
        assert graph.get_reachable_nodes("A", "o>") == {"A", "I"}
        assert graph.get_reachable_nodes("A", "<o") == {"A", "J"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

    def test_get_reachable_nodes_fails(self):
        """Test failing `get_reachable_nodes` method of the `_CoreGraph` class"""
        # Test1: The `_CoreGraph` do not have any nodes.
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_reachable_nodes("A", "->")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # Test2: The `_CoreGraph` have nodes. But, do not have any edges.
        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_reachable_nodes("A", "->") == {"A"}
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        # Test3: Wrong input values.
        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_reachable_nodes()
        with pytest.raises(TypeError):
            graph.get_reachable_nodes("A")

        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_from_api_edge_type_cases(self):
        # 1. Circle-Line Edge
        edge_tuple = ("A", "B", "o-")
        graph = _CoreGraph()
        assert graph._from_api_edge_type(edge_tuple) == {"B": "-", "A": "o"}

        # 2. Arrow-Circle Edge
        edge_tuple = ("A", "B", "<o")
        graph = _CoreGraph()
        assert graph._from_api_edge_type(edge_tuple) == {"B": "o", "A": ">"}
        # 3. Bidirected Edge
        edge_tuple = ("A", "B", "<>")
        graph = _CoreGraph()
        assert graph._from_api_edge_type(edge_tuple) == {"A": ">", "B": ">"}

        # 4. Undirected Edge (General case via else block)
        edge_tuple = ("A", "B", "--")
        graph = _CoreGraph()
        assert graph._from_api_edge_type(edge_tuple) == {"A": "-", "B": "-"}

        # 5. Directed Edge (Forward - General case via else block)
        edge_tuple = ("A", "B", "->")
        graph = _CoreGraph()
        assert graph._from_api_edge_type(edge_tuple) == {"A": "-", "B": ">"}

    def test_from_api_edge_type_fails(self):
        invalid_edge = ("A", "B", "key", "<-")
        graph = _CoreGraph()
        with pytest.raises(ValueError):
            graph._from_api_edge_type(invalid_edge)

    def test_to_api_edge_type_cases(self):
        # 1. Reverse Arrow Edge (Explicit check: u='>', v='-')
        u, v = "B", "A"
        markers = {"B": ">", "A": "-"}
        graph = _CoreGraph()
        assert graph._to_api_edge_type(u, v, markers) == "<-"

        # 2. Circle-Line Edge (Explicit check: u='o', v='-')
        u, v = "A", "B"
        markers = {"A": "o", "B": "-"}
        graph = _CoreGraph()
        assert graph._to_api_edge_type(u, v, markers) == "o-"

        # 3. Arrow-Circle Edge (Explicit check: u='>', v='o')
        u, v = "A", "B"
        markers = {"A": ">", "B": "o"}
        graph = _CoreGraph()
        assert graph._to_api_edge_type(u, v, markers) == "<o"

        # 4. Bidirected Edge (Explicit check: u='>', v='>')
        u, v = "A", "B"
        markers = {"A": ">", "B": ">"}
        graph = _CoreGraph()
        assert graph._to_api_edge_type(u, v, markers) == "<>"

        # 5. Directed Edge (Forward - General case via else block)
        u, v = "A", "B"
        markers = {"A": "-", "B": ">"}
        graph = _CoreGraph()
        assert graph._to_api_edge_type(u, v, markers) == "->"

        # 6. Undirected Edge (General case via else block)
        u, v = "A", "B"
        markers = {"A": "-", "B": "-"}
        graph = _CoreGraph()

        assert graph._to_api_edge_type(u, v, markers) == "--"

    def test_to_api_edge_type_fails(self):
        # Missing marker for node 'u' or 'v'
        u, v = "A", "B"
        markers = {"A": "-"}  # 'B' is missing
        graph = _CoreGraph()

        with pytest.raises(KeyError):
            graph._to_api_edge_type(u, v, markers)

    def test_get_edges(self):
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "o-")
        graph.add_edge("A", "B", "<>")

        assert sorted(graph.get_edges(keys=False, data=False)) == sorted(
            [
                ("A", "B"),
                ("A", "B"),
                ("B", "C"),
            ]
        )
        assert sorted(graph.get_edges(keys=True, data=False)) == sorted(
            [
                ("A", "B", 0),
                ("A", "B", 1),
                ("B", "C", 0),
            ]
        )
        assert sorted(graph.get_edges(keys=False, data=True)) == sorted(
            [
                ("A", "B", "->"),
                ("A", "B", "<>"),
                ("B", "C", "o-"),
            ]
        )
        assert sorted(graph.get_edges(keys=True, data=True)) == sorted(
            [
                ("A", "B", 0, "->"),
                ("A", "B", 1, "<>"),
                ("B", "C", 0, "o-"),
            ]
        )

    def test_get_edge_type(self):
        graph = _CoreGraph()
        assert (
            set(["--", "-o", "o-", "->", "<-", "o>", "<o", "<>", "oo"])
            == graph.get_edge_type()
        )
