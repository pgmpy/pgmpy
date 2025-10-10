import pytest

from pgmpy.base import PAG


@pytest.fixture
def pag():
    edges = [
        ("A", "B", "-", ">"),
        ("A", "C", ">", "-"),
        ("B", "C", "-", "-"),
        ("D", "E", "o", "o"),
        ("E", "F", "o", "o"),
        ("U", "W", "o", ">"),
        ("V", "W", ">", "-"),
        ("X", "W", ">", "-"),
        ("U", "V", "-", "-"),
        ("U", "X", "-", "-"),
        ("M", "N", "-", ">"),
        ("N", "O", "-", ">"),
    ]
    return PAG(ebunch=edges)


class TestPAGRules:
    all_rules = [f"rule_{i}" for i in range(1, 11)]
    # list of all the rules

    def test_rule_1(self):
        pag = PAG(
            ebunch=[
                ("a", "b", "-", "o"),
                ("a", "d", "-", "-"),
                ("b", "c", "o", "-"),
            ]
        )
        expected_pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),
                ("a", "d", "-", "-"),
                ("b", "c", ">", "-"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_1":
                pag_new = pag.rule_1(separating_sets=None)
                assert pag_new == expected_pag

            else:
                pag_new = getattr(pag, rule)(separating_sets=None)
                assert pag_new == pag

    def test_rule_2(self):
        pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),
                ("b", "c", "-", "o"),
                ("b", "d", "-", "o"),
            ]
        )
        expected_pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),
                ("b", "c", "-", ">"),
                ("b", "d", "-", "o"),
            ]
        )
        for rule in self.all_rules:
            if rule == "rule_2":
                pag_new = pag.rule_2(separating_sets=None)
                assert pag_new == expected_pag

            else:
                pag_new = getattr(pag, rule)(separating_sets=None)
                assert pag_new == pag

    def test_rule_3(self):
        pass

    def test_rule_4(self):
        pass

    def test_rule_5(self):
        pag = PAG(
            ebunch=[
                ("a", "b", "o", "o"),
                ("a", "c", "o", "o"),
                ("c", "d", "o", "o"),
                ("d", "e", "o", "o"),
                ("e", "b", "o", "o"),
            ]
        )
        expected_pag = PAG(
            ebunch=[
                ("a", "b", "-", "-"),
                ("a", "c", "-", "-"),
                ("c", "d", "-", "-"),
                ("d", "e", "-", "-"),
                ("e", "b", "-", "-"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_5":
                pag_new = pag.rule_5(separating_sets=None)
                assert pag_new == expected_pag

            else:
                pag_new = getattr(pag, rule)(separating_sets=None)
                assert pag_new == pag

    def test_rule_5_edge_case(self):
        pag = PAG(
            ebunch=[
                ("a", "b", "-", "-"),
                ("a", "c", "-", "-"),
                ("c", "d", "-", "-"),
                ("d", "e", "-", "-"),
                ("e", "b", "-", "-"),
                ("c", "b", "o", "o"),
            ]
        )
        pag_new = pag.rule_5(separating_sets=None)
        assert pag == pag_new

    def test_rule_6(self):
        pag = PAG(
            ebunch=[
                ("u", "v", "-", "-"),
                ("v", "w", "o", "o"),
            ]
        )
        expected_pag = PAG(
            ebunch=[
                ("u", "v", "-", "-"),
                ("v", "w", ">", "-"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_6":
                pag_new = pag.rule_6(separating_sets=None)
                assert pag_new == expected_pag

            else:
                pag_new = getattr(pag, rule)(separating_sets=None)
                assert pag_new == pag

    def test_rule_7(self):
        pag = PAG(
            ebunch=[
                ("u", "v", "o", "-"),
                ("v", "w", "o", "-"),
                ("u", "x", "o", "o"),
                ("x", "v", "-", "-"),
            ]
        )

        expected_pag = PAG(
            ebunch=[
                ("u", "v", "o", "-"),
                ("v", "w", "o", "-"),
                ("u", "x", "o", "o"),
                ("x", "v", ">", "-"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_7":
                pag_new = pag.rule_7(separating_sets=None)
                assert pag_new == expected_pag

            else:
                pag_new = getattr(pag, rule)(separating_sets=None)
                assert pag_new == pag

    def test_rule_8(self):
        pag = PAG(
            ebunch=[
                ("u", "v", "-", ">"),
                ("v", "w", "-", ">"),
                ("u", "w", "o", ">"),
            ]
        )

        expected_pag = PAG(
            ebunch=[
                ("u", "v", "-", ">"),
                ("v", "w", "-", ">"),
                ("u", "w", "-", ">"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_8":
                pag_new = pag.rule_8(separating_sets=None)
                assert pag_new == expected_pag

            else:
                pag_new = getattr(pag, rule)(separating_sets=None)
                assert pag_new == pag

    def test_rule_9(self):
        pag = PAG(
            ebunch=[
                ("a", "b", "o", "o"),
                ("a", "c", "o", "o"),
                ("b", "d", "o", "o"),
                ("c", "d", "o", "o"),
            ]
        )

        expected_pag = PAG(
            ebunch=[
                ("a", "b", "o", "o"),
                ("a", "c", "o", "o"),
                ("b", "d", "-", ">"),
                ("c", "d", "-", ">"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_9":
                pag_new = pag.rule_9(separating_sets=None)
                assert pag_new == expected_pag

            else:
                pag_new = getattr(pag, rule)(separating_sets=None)
                assert pag_new == pag

    def test_rule_10(self):
        pass


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
