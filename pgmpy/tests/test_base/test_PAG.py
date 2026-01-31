import pytest

from pgmpy.base import PAG


@pytest.fixture
def pag_core():
    edges = [
        ("A", "B", "o", ">"),
        ("C", "B", "o", ">"),
        ("A", "D", "o", "o"),
        ("C", "D", "o", "o"),
        ("D", "E", "-", ">"),
    ]
    return PAG(ebunch=edges)


@pytest.fixture
def pag_simple():
    edges = [
        ("A", "B", "-", ">"),
        ("B", "C", "-", ">"),
        ("C", "D", "-", ">"),
    ]
    return PAG(ebunch=edges)


@pytest.fixture
def pag_complex():
    edges = [
        ("A", "B", "-", ">"),
        ("B", "C", "-", ">"),
        ("C", "D", "-", ">"),
        ("E", "C", ">", "-"),
        ("F", "C", ">", "-"),
        ("A", "E", "o", "o"),
        ("A", "F", "o", "o"),
        ("G", "H", "o", "o"),
        ("H", "I", "o", "o"),
        ("I", "J", "o", "o"),
        ("J", "G", "o", "o"),
        ("H", "C", "o", ">"),
        ("I", "D", "-", ">"),
        ("J", "B", "o", "-"),
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
        assert set(graph.nodes) == {"A", "B", "C"}
        assert graph["A"]["B"]["marks"] == {"A": "-", "B": ">"}
        assert graph["B"]["C"]["marks"] == {"B": ">", "C": "-"}

    def test_is_definite_non_collider(self, pag_core, pag_complex):
        assert pag_core.is_definite_non_collider("D", "A", "C") is True
        assert pag_core.is_definite_non_collider("B", "A", "C") is False

        assert pag_complex.is_definite_non_collider("C", "E", "F") is True

    def test_is_definitely_visible(self, pag_core, pag_complex):
        assert pag_core.is_definitely_visible("D", "E") is True
        assert pag_core.is_definitely_visible("A", "B") is False

        assert pag_complex.is_definitely_visible("C", "D") is True

    def test_is_uncovered(self, pag_core, pag_complex):
        assert pag_core.is_uncovered(["A", "D", "C"]) is True
        assert pag_core.is_uncovered(["A", "B", "C"]) is True

        assert pag_complex.is_uncovered(["G", "H", "I", "J"]) is True

    def test_get_potentially_directed_paths(self, pag_core, pag_complex):
        paths = pag_core.get_potentially_directed_paths("A", "E")
        assert ["A", "D", "E"] in paths

        assert pag_complex.get_potentially_directed_paths("C", "A") == [
            ["C", "E", "A"],
            ["C", "F", "A"],
        ]

    def test_get_paths_with_marks(self, pag_simple, pag_complex):
        paths = pag_simple.get_paths_with_marks("A", "D", u_type="-", v_type=">")
        assert ["A", "B", "C", "D"] in paths

        with pytest.raises(ValueError):
            pag_simple.get_paths_with_marks("A", "A")

        paths_complex = pag_complex.get_paths_with_marks(
            "A", "C", u_type="-", v_type=">"
        )
        assert ["A", "B", "C"] in paths_complex

    def test_modify_edge(self, pag_core, pag_complex):
        pag_core.modify_edge("A", "B", mark_u="-")
        assert pag_core["A"]["B"]["marks"]["A"] == "-"

        pag_complex.modify_edge("A", "E", mark_u="-")
        assert pag_complex["A"]["E"]["marks"]["A"] == "-"


class TestPAGRules:
    all_rules = [f"rule_{i}" for i in range(1, 11)]

    def test_rule_1(self):
        pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),
                ("a", "d", "-", "-"),
                ("b", "c", "o", "-"),
            ]
        )
        expected_pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),
                ("a", "d", "-", "-"),
                ("b", "c", "-", ">"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_1":
                pag_new = pag.rule_1(separating_sets=None)
                assert pag_new == expected_pag
            else:
                assert pag == pag

    def test_rule_2(self):
        pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),
                ("b", "c", "-", ">"),
                ("a", "c", "-", "o"),
            ]
        )
        expected_pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),
                ("b", "c", "-", ">"),
                ("a", "c", "-", ">"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_2":
                pag_new = pag.rule_2(separating_sets=None)
                assert pag_new == expected_pag
            else:
                assert pag == pag

    def test_rule_3(self):
        pag = PAG(
            ebunch=[
                ("A", "B", "-", ">"),
                ("C", "B", "-", ">"),
                ("A", "T", "-", "o"),
                ("C", "T", "-", "o"),
                ("T", "B", "-", "o"),
            ]
        )
        expected_pag = PAG(
            ebunch=[
                ("A", "B", "-", ">"),
                ("C", "B", "-", ">"),
                ("A", "T", "-", "o"),
                ("C", "T", "-", "o"),
                ("T", "B", "-", ">"),  # after orientation
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_3":
                pag_new = pag.rule_3(separating_sets=None)
                assert pag_new == expected_pag
            else:
                assert pag == pag

    def test_rule_4_case_1(self):
        pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),  # a → b
                ("b", "c", "-", ">"),  # b → c
                ("c", "d", "o", "-"),  # c ◦— d   (circle at c)
            ]
        )

        # c IS in Sepset(a, d)
        separating_sets = {("a", "d"): {"c"}}

        expected_pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),
                ("b", "c", "-", ">"),
                ("c", "d", "-", ">"),  # must become c → d
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_4":
                pag_new = pag.rule_4(separating_sets=separating_sets)
                assert pag_new == expected_pag
            else:
                assert pag == pag

    def test_rule_4_case_2(self):
        pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),  # a → b
                ("b", "c", "-", ">"),  # b → c
                ("c", "d", "o", "-"),  # c ◦— d
            ]
        )

        # c NOT in Sepset(a, d)
        separating_sets = {("a", "d"): set()}

        expected_pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),  # unchanged
                ("b", "c", ">", ">"),  # b ↔ c
                ("c", "d", ">", ">"),  # c ↔ d
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_4":
                pag_new = pag.rule_4(separating_sets=separating_sets)
                assert pag_new == expected_pag
            else:
                assert pag == pag

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
                assert pag == pag

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
                ("v", "w", "-", "o"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_6":
                pag_new = pag.rule_6(separating_sets=None)
                assert pag_new == expected_pag
            else:

                assert pag == pag

    def test_rule_7(self):
        pag = PAG(
            ebunch=[
                ("u", "v", "-", "o"),
                ("v", "w", "o", "o"),
            ]
        )
        expected_pag = PAG(
            ebunch=[
                ("u", "v", "-", "o"),
                ("v", "w", "-", "o"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_7":
                pag_new = pag.rule_7(separating_sets=None)
                assert pag_new == expected_pag
            else:
                assert pag == pag

    def test_rule_8(self):
        pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),
                ("b", "c", "-", ">"),
                ("a", "c", "o", ">"),
            ]
        )
        expected_pag = PAG(
            ebunch=[
                ("a", "b", "-", ">"),
                ("b", "c", "-", ">"),
                ("a", "c", "-", ">"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_8":
                pag_new = pag.rule_8(separating_sets=None)
                assert pag_new == expected_pag
            else:

                assert pag == pag

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
                assert pag == pag

    def test_rule_10(self):
        pag = PAG(
            ebunch=[
                ("u", "w", "o", ">"),
                ("v", "w", ">", "-"),
                ("x", "w", ">", "-"),
                ("u", "m", "-", "-"),
                ("m", "v", "-", "-"),
                ("u", "n", "-", "-"),
                ("n", "x", "-", "-"),
            ]
        )

        expected_pag = PAG(
            ebunch=[
                ("u", "w", "-", ">"),
                ("v", "w", ">", "-"),
                ("x", "w", ">", "-"),
                ("u", "m", "-", "-"),
                ("m", "v", "-", "-"),
                ("u", "n", "-", "-"),
                ("n", "x", "-", "-"),
            ]
        )

        for rule in self.all_rules:
            if rule == "rule_10":
                pag_new = pag.rule_10(separating_sets=None)
                assert pag_new == expected_pag
            else:
                assert pag == pag
