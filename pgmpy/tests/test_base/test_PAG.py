import pytest

from pgmpy.base import PAG


class TestPAG:
    def test_init_and_edge_marks(self):
        pag = PAG(edge_list=[("A", "B", "o>"), ("B", "C", "oo")])

        assert pag.get_edge_marks("A", "B") == {"A": "o", "B": ">"}
        assert pag.get_edge_marks("B", "C") == {"B": "o", "C": "o"}

    def test_modify_edge(self):
        pag = PAG(edge_list=[("A", "B", "oo")])

        pag.modify_edge("A", "B", mark_u="-", mark_v=">")

        assert pag.get_edge_type("A", "B") == {"->"}

    def test_rejects_parallel_edges(self):
        pag = PAG(edge_list=[("A", "B", "oo")])

        with pytest.raises(ValueError, match="already has an edge"):
            pag.add_edge("A", "B", "->")

    def test_definite_non_collider(self):
        pag = PAG(edge_list=[("A", "B", "oo"), ("B", "C", "oo")])
        assert pag.is_definite_non_collider("A", "B", "C")

        pag.add_edge("A", "C", "oo")
        assert not pag.is_definite_non_collider("A", "B", "C")

        pag.modify_edge("A", "B", mark_v="-")
        assert pag.is_definite_non_collider("A", "B", "C")

    def test_is_uncovered(self):
        pag = PAG(edge_list=[("A", "B", "oo"), ("B", "C", "o>")])

        assert pag.is_uncovered(["A", "B", "C"])

        pag.add_edge("A", "C", "oo")
        assert not pag.is_uncovered(["A", "B", "C"])

    def test_is_uncovered_rejects_invalid_path(self):
        pag = PAG(edge_list=[("A", "B", "oo")])

        with pytest.raises(ValueError, match="at least two"):
            pag.is_uncovered(["A"])
        with pytest.raises(ValueError, match="consecutive pair"):
            pag.is_uncovered(["A", "B", "C"])

    def test_potentially_directed_paths_and_possible_ancestors(self):
        pag = PAG(
            edge_list=[
                ("A", "B", "o>"),
                ("B", "C", "->"),
                ("D", "C", "<>"),
            ]
        )

        assert pag.get_potentially_directed_paths("A", "C") == [["A", "B", "C"]]
        assert pag.get_potentially_directed_paths("C", "A") == []
        assert pag.get_possible_ancestors("C") == {"A", "B", "C"}

    def test_potentially_directed_paths_reject_invalid_endpoints(self):
        pag = PAG(edge_list=[("A", "B", "oo")])

        with pytest.raises(ValueError, match="must differ"):
            pag.get_potentially_directed_paths("A", "A")
        with pytest.raises(ValueError, match="must both be present"):
            pag.get_potentially_directed_paths("A", "C")
