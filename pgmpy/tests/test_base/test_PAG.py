import pytest

from pgmpy.base import PAG


@pytest.fixture
def pag():
    edges = [
        # Simple directed and bidirected cases
        ("A", "B", "-", ">"),
        ("A", "C", ">", "-"),
        ("B", "C", "-", "-"),  # to test uncovered vs. covered
        # Circle edges
        ("D", "E", "o", "o"),
        ("E", "F", "o", "o"),
        # Potential fork setup
        ("U", "W", "o", ">"),
        ("V", "W", ">", "-"),
        ("X", "W", ">", "-"),
        ("U", "V", "-", "-"),
        ("U", "X", "-", "-"),
        # Another path with arrows
        ("M", "N", "-", ">"),
        ("N", "O", "-", ">"),
    ]
    return PAG(ebunch=edges)


class TestPAG:
    def test_init_empty(self):
        graph = PAG()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_init_with_edges(self):
        edges = [("A", "B", "-", ">"), ("B", "C", ">", "-")]
        graph = PAG(ebunch=edges)
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph["A"]["B"]["marks"] == {"A": "-", "B": ">"}
        assert graph["B"]["C"]["marks"] == {"B": ">", "C": "-"}

    def test_is_definitely_a_non_collider(self, pag):
        assert pag.is_definite_non_collider("A", "B", "C") is True
        assert pag.is_definite_non_collider("E", "D", "F") is True

    def test_possible_ancestors(self, pag):
        assert "A" in pag.get_possible_ancestors("A")
        ancestors_B = pag.get_possible_ancestors("B")
        assert {"A", "B"} <= ancestors_B
        assert "U" in pag.get_possible_ancestors("V")

    def test_is_possibly_visible(self, pag):
        assert pag.is_definitely_visible("A", "B") is True
        assert pag.is_definitely_visible("A", "C") is False
        assert pag.is_definitely_visible("X", "Z") is False

    def test_is_uncovered(self, pag):
        assert pag.is_uncovered(["B", "A", "C"]) is False
        assert pag.is_uncovered(["D", "E", "F"]) is True

    def test_get_potentially_directed(self, pag):
        paths = pag.get_potentially_directed("A", "C")
        assert ["A", "C"] in paths or ["A", "B", "C"] in paths
        paths = pag.get_potentially_directed("M", "O")
        assert ["M", "N", "O"] in paths
        assert pag.get_potentially_directed("B", "Y") == []

    def is_valid_fork_configuration(self, pag):
        assert pag.is_valid_fork_configuration("U", "W", forks=["V", "X"]) is True

    def test_get_path_with_marks(self, pag):
        paths = pag.get_paths_with_marks("A", "B", u_type="-", v_type=">")
        assert ["A", "B"] in paths
        assert pag.get_paths_with_marks("A", "C", u_type="-", v_type=">") == []

    def test_modify_edge(self, pag):
        pag.add_edge("A", "B", "-", ">")
        assert pag["A"]["B"]["marks"] == {"A": "-", "B": ">"}
        pag.modify_edge("A", "B", mark_u="o", mark_v="o")
        assert pag["A"]["B"]["marks"] == {"A": "o", "B": "o"}

    # Defining the test functions for rules

    def test_rule_1(self):
        pag = PAG()
        pag = pag.rule_1(pag)
        expected_pag = PAG()

        assert pag == expected_pag
        # then you are supposed to make sure that the graph does not change when we apply other rules

    def test_rule_2(self):
        pag = PAG()
        pag = pag.rule_2(pag)
        expected_pag = PAG()

        assert pag == expected_pag

    def test_rule_3(self):
        pass

    def test_rule_4(self):
        pass

    def test_rule_5(self):
        pass

    # def test_rule_6(self):
    #     edges = [
    #         ("u", "v", "-", "-"),
    #         ("v", "w", "o", "o"),
    #     ]

    #     pag = PAG(ebunch=edges)
    #     pag_after_rule = pag.rule_6()

    #     edges_required = [
    #         ("u", "v", "-", "-"),
    #         ("v", "w", ">", "-"),
    #     ]

    #     pag_new = PAG(ebunch=edges_required)

    #     assert pag_after_rule == pag_new

    # def test_rule_7(self):
    #     edges = [
    #         ("u", "v", "o", "-"),
    #         ("v", "w", "o", "-"),
    #         ("u", "x", "o", "o"),
    #         ("x", "v", "-", "-"),
    #     ]

    #     pag = PAG(ebunch=edges)

    #     pag_after_rule = pag.rule_7()

    #     edges_required = [
    #         ("u", "v", "o", "-"),
    #         ("v", "w", "o", "-"),
    #         ("u", "x", "o", "o"),
    #         ("x", "v", ">", "-"),
    #     ]

    #     pag_new = PAG(ebunch=edges_required)

    #     assert pag_after_rule == pag_new

    def test_rule_8(self):
        edges = [
            ("u", "v", "-", ">"),
            ("v", "w", "-", ">"),
            ("u", "w", "o", ">"),
        ]

        pag = PAG(ebunch=edges)

        pag_after_rule = pag.rule_8()

        edges_required = [
            ("u", "v", "-", ">"),
            ("v", "w", "-", ">"),
            ("u", "w", "-", ">"),
        ]

        pag_new = PAG(edges_required)

        assert pag_after_rule == pag_new

    # def test_rule_9(self):
    #     edges = [
    #         ("a", "b", "o", "o"),
    #         ("a", "c", "o", "o"),
    #         ("b", "d", "o", "o"),
    #         ("c", "d", "o", "o"),
    #     ]

    #     pag = PAG(ebunch=edges)

    #     pag_after_rule = pag.rule_9(pag)
    #     pag_after_rule = pag_after_rule.rule_9(pag_after_rule)

    #     edges_required = [
    #         ("a", "b", "o", "o"),
    #         ("a", "c", "o", "o"),
    #         ("b", "d", "-", ">"),
    #         ("c", "d", "-", ">"),
    #     ]
    #     pag_new = PAG(edges_required)

    #     assert pag_after_rule == pag_new

    def test_rule_10(self):
        pass
