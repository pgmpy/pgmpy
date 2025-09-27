import pytest

from pgmpy.base import PAG


@pytest.fixture
def pag():
    edges = [
        (),
    ]

    pag = PAG(ebunch=edges)

    return pag


class TestPAG(PAG):
    def __init__(self):
        pass

    def test_is_definitely_a_non_collider(self):
        pass

    def test_possible_ancestors(self):
        pass

    def test_is_possibly_visible(self):
        pass

    def test_is_uncovered(self):
        pass

    def test_get_potentially_directed(self):
        pass

    def is_valid_fork_configuration(self):
        pass

    def test_get_path_with_marks(self):
        pass

    def test_modify_edge(self):
        pass

    # Defining the test functions for rules

    def test_rule_1(self):
        pag = PAG()
        pag = self.rule_1(pag)
        expected_pag = PAG()

        assert pag == expected_pag
        changed = True
        for i in range(1, 11):
            currRule = f"rule_{i}"
            if i == 1:
                continue
            pag_new = currRule(pag)
            if pag_new != pag:
                print(f"Rule {i} modifies the graph which is problematic")
                changed = False
                break

        assert changed == True
        # then you are supposed to make sure that the graph does not change when we apply other rules

    def test_rule_2(self):
        pag = PAG()
        pag = self.rule_2(pag)
        expected_pag = PAG()

        assert pag == expected_pag

    def test_rule_3(self):
        pass

    def test_rule_4(self):
        pass

    def test_rule_5(self):
        pass

    def test_rule_6(self):
        edges = [
            ("u", "v", "-", "-"),
            ("v", "w", "o", None),
        ]

        pag = PAG(ebunch=edges)
        # Apply the rule
        pag_after_rule = self.rule_6(pag)

        edges_required = [("u", "v", "-", "-"), ("v", "w", "-", None)]

        pag_new = PAG(ebunch=edges_required)

        # Check if equal
        assert pag_after_rule == pag_new

    def test_rule_7(self):
        edges = [
            ("u", "v", "-", "o"),
            ("v", "w", "o", None),
        ]

        pag = PAG(ebunch=edges)

        pag_after_rule = self.rule_7(pag)

        edges_required = [
            ("u", "v", "-", "o"),
            ("v", "w", "-", None),
        ]

        pag_new = PAG(edges_required)

        assert pag_after_rule == pag_new

    def test_rule_8(self):
        edges = [
            ("u", "v", "-", ">"),
            ("v", "w", "-", ">"),
            ("u", "w", "o", ">"),
        ]

        pag = PAG(ebunch=edges)

        pag_after_rule = self.rule_8(pag)

        edges_required = [
            ("u", "v", "-", ">"),
            ("v", "w", "-", ">"),
            ("u", "w", "-", ">"),
        ]

        pag_new = PAG(edges_required)

        assert pag_after_rule == pag_new

    def test_rule_9(self):
        edges = [
            ("a", "b", "o", "o"),
            ("a", "c", "o", "o"),
            ("b", "d", "o", ">"),
            ("c", "d", "o", ">"),
        ]

        pag = PAG(ebunch=edges)

        pag_after_rule = self.rule_9(pag)
        pag_after_rule = self.rule_9(pag_after_rule)

        edges_required = [
            ("a", "b", "o", "o"),
            ("a", "c", "o", "o"),
            ("b", "d", "-", ">"),
            ("c", "d", "-", ">"),
        ]
        pag_new = PAG(edges_required)

        assert pag_after_rule == pag_new

    def test_rule_10(self):
        pass

    def test_is_equal(self, other):
        pass
