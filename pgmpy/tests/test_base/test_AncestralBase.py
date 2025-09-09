import numpy as np
import pytest

from pgmpy.base import AncestralBase


@pytest.fixture
def base_graph():
    edges = [
        ("A", "B", "-", ">"),
        ("A", "C", ">", "-"),
        ("A", "D", "o", "o"),
        ("A", "E", ">", ">"),
        ("A", "F", "-", "-"),
        ("A", "G", "-", "o"),
        ("A", "H", "o", "-"),
        ("A", "I", "o", ">"),
        ("A", "J", ">", "o"),
        ("B", "X", "-", ">"),
        ("C", "Y", ">", "-"),
    ]
    return AncestralBase(ebunch=edges)


class TestAncestralBase:
    def test_init_empty(self):
        graph = AncestralBase()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_init_with_edges(self):
        edges = [("A", "B", "-", ">"), ("B", "C", ">", "-")]
        graph = AncestralBase(ebunch=edges)
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph["A"]["B"]["marks"] == {"A": "-", "B": ">"}
        assert graph["B"]["C"]["marks"] == {"B": ">", "C": "-"}

    def test_add_edge_same_node_error(self, base_graph):
        with pytest.raises(ValueError):
            base_graph.add_edge("A", "A", "-", ">")

    def test_add_edge_invalid_marks_error(self, base_graph):
        with pytest.raises(ValueError):
            base_graph.add_edge("A", "B", "x", ">")
        with pytest.raises(ValueError):
            base_graph.add_edge("A", "B", "-", "y")
        with pytest.raises(ValueError):
            base_graph.add_edge("A", "B", "z", "w")

    def test_add_edges_from(self):
        graph = AncestralBase()
        edges = [("A", "B", "-", ">"), ("B", "C", ">", "-"), ("A", "C", "o", "o")]
        graph.add_edges_from(edges)

        assert len(graph.edges) == 3
        assert len(graph.nodes) == 3
        assert graph["A"]["B"]["marks"] == {"A": "-", "B": ">"}
        assert graph["B"]["C"]["marks"] == {"B": ">", "C": "-"}
        assert graph["A"]["C"]["marks"] == {"A": "o", "C": "o"}

    def test_get_neighbors_basic(self, base_graph):
        all_nodes_except_A = {"B", "C", "D", "E", "F", "G", "H", "I", "J"}
        assert base_graph.get_neighbors("A") == all_nodes_except_A
        assert base_graph.get_neighbors("B") == {"A", "X"}
        assert base_graph.get_neighbors("C") == {"A", "Y"}

    def test_get_neighbors_nonexistent_node(self, base_graph):
        assert base_graph.get_neighbors("Z") == set()

    def test_get_parents(self, base_graph):
        assert base_graph.get_parents("A") == {"C"}
        assert base_graph.get_parents("B") == {"A"}
        assert base_graph.get_parents("C") == {"Y"}

    def test_get_children(self, base_graph):
        assert base_graph.get_children("A") == {"B"}
        assert base_graph.get_children("B") == {"X"}
        assert base_graph.get_children("Y") == {"C"}

    def test_get_spouses(self, base_graph):
        assert base_graph.get_spouses("A") == {"E"}
        assert base_graph.get_spouses("E") == {"A"}
        assert base_graph.get_spouses("B") == set()

    def test_get_ancestors(self, base_graph):
        assert base_graph.get_ancestors("A") == {"A", "C", "Y"}
        assert base_graph.get_ancestors("X") == {"X", "B", "A", "C", "Y"}

    def test_get_descendants(self, base_graph):
        assert base_graph.get_descendants("A") == {"A", "B", "X"}
        assert base_graph.get_descendants("Y") == {"Y", "C", "A", "B", "X"}

    def test_get_reachable_nodes(self, base_graph):
        assert base_graph.get_reachable_nodes("A", v_type=">") == {
            "A",
            "B",
            "X",
            "E",
            "I",
        }
        assert base_graph.get_reachable_nodes("A", u_type=">", v_type="-") == {
            "A",
            "C",
            "Y",
        }
        assert base_graph.get_reachable_nodes("A", u_type="o", v_type="o") == {"A", "D"}
        assert base_graph.get_reachable_nodes("A", u_type=">", v_type=">") == {"A", "E"}
        assert base_graph.get_reachable_nodes("A", u_type="-", v_type="-") == {"A", "F"}
        assert base_graph.get_reachable_nodes("A", u_type="o", v_type=">") == {"A", "I"}
        assert base_graph.get_reachable_nodes("A", u_type=">", v_type="o") == {"A", "J"}

    def test_adjacency_matrix(self):
        edges = [("A", "B", "-", ">"), ("B", "C", ">", "-")]
        graph = AncestralBase(ebunch=edges)
        M, node_index = graph.adjacency_matrix

        expected = np.array([[0, ">", 0], ["-", 0, "-"], [0, ">", 0]], dtype=object)

        assert M.shape == (3, 3)
        assert len(node_index) == 3
        assert expected.tolist() == M.tolist()
        assert set(node_index.keys()) == {"A", "B", "C"}

    def test_adjacency_matrix_empty_graph(self):
        graph = AncestralBase()
        M, node_index = graph.adjacency_matrix
        assert M.shape == (0, 0)
        assert len(node_index) == 0

    def test_adjacency_matrix_setter(self):
        M = np.array([[0, ">", 0], ["-", 0, ">"], [0, "-", 0]], dtype=object)
        graph = AncestralBase()
        graph.adjacency_matrix = M

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.has_edge("X_0", "X_1")
        assert graph.has_edge("X_1", "X_2")
        assert graph["X_0"]["X_1"]["marks"] == {"X_0": ">", "X_1": "-"}
        assert graph["X_1"]["X_2"]["marks"] == {"X_1": ">", "X_2": "-"}

    def test_init_with_roles(self):
        edges = [("A", "B", "-", ">"), ("B", "C", ">", "-")]
        roles = {"exposure": "A", "outcome": "C"}
        graph = AncestralBase(ebunch=edges, roles=roles)

        assert graph.nodes["A"]["role"] == "exposure"
        assert graph.nodes["C"]["role"] == "outcome"
        assert "role" not in graph.nodes["B"]

    def test_with_role_method(self):
        graph = AncestralBase([("A", "B", "-", ">")])
        graph = graph.with_role("instrument", "A")
        assert graph.nodes["A"]["role"] == "instrument"

        graph = graph.with_role("adjustment", {"A", "B"}, inplace=True)
        assert graph.nodes["A"]["role"] == "adjustment"
        assert graph.nodes["B"]["role"] == "adjustment"

    def test_copy_preserves_roles(self):
        edges = [("A", "B", "-", ">"), ("A", "C", "-", ">")]
        roles = {"exposure": "A", "outcome": "B"}
        graph = AncestralBase(ebunch=edges, roles=roles)
        new_graph = graph.copy()

        assert new_graph.nodes["A"]["role"] == "exposure"
        assert new_graph.nodes["B"]["role"] == "outcome"

    def test_equality_with_roles(self):
        edges = [("A", "B", "-", ">")]
        roles = {"exposure": "A"}
        g1 = AncestralBase(edges, roles=roles)
        g2 = AncestralBase(edges, roles=roles)
        g3 = AncestralBase(edges, roles={"outcome": "A"})

        assert g1 == g2
        assert g1 != g3
