import pytest

from pgmpy.base import DAG, PDAG


class TestPDAG:
    def test_init_normal(self):
        # Mix directed and undirected
        ebunch_mix = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag = PDAG(edge_list=ebunch_mix)
        assert set(pdag.nodes()) == {"A", "B", "C", "D"}
        assert sorted(pdag.get_edges(data=True)) == [
            ("A", "B", "--"),
            ("A", "C", "->"),
            ("D", "B", "--"),
            ("D", "C", "->"),
        ]

        pdag = PDAG(
            edge_list=ebunch_mix,
            latents=["A", "C"],
        )
        assert set(pdag.nodes()) == {"A", "B", "C", "D"}
        assert sorted(pdag.get_edges(data=True)) == [
            ("A", "B", "--"),
            ("A", "C", "->"),
            ("D", "B", "--"),
            ("D", "C", "->"),
        ]
        assert pdag.latents == {"A", "C"}

        # Only undirected
        ebunch_undir = [("A", "C", "--"), ("D", "C", "--"), ("B", "A", "--"), ("B", "D", "--")]
        pdag = PDAG(edge_list=ebunch_undir)
        assert set(pdag.nodes()) == {"A", "B", "C", "D"}
        assert sorted(pdag.get_edges(data=True)) == [
            ("A", "B", "--"),
            ("A", "C", "--"),
            ("C", "D", "--"),
            ("D", "B", "--"),
        ]

        pdag = PDAG(edge_list=ebunch_undir, latents=["A", "D"])
        assert set(pdag.nodes()) == {"A", "B", "C", "D"}
        assert sorted(pdag.get_edges(data=True)) == [
            ("A", "B", "--"),
            ("A", "C", "--"),
            ("C", "D", "--"),
            ("D", "B", "--"),
        ]
        assert pdag.latents == {"A", "D"}

        # Only directed
        ebunch_dir = [("A", "B", "->"), ("D", "B", "->"), ("A", "C", "->"), ("D", "C", "->")]
        pdag = PDAG(edge_list=ebunch_dir)
        assert set(pdag.nodes()) == {"A", "B", "C", "D"}
        assert sorted(pdag.get_edges(data=True)) == [
            ("A", "B", "->"),
            ("A", "C", "->"),
            ("D", "B", "->"),
            ("D", "C", "->"),
        ]

        pdag = PDAG(edge_list=ebunch_dir, latents=["D"])
        assert set(pdag.nodes()) == {"A", "B", "C", "D"}
        assert sorted(pdag.get_edges(data=True)) == [
            ("A", "B", "->"),
            ("A", "C", "->"),
            ("D", "B", "->"),
            ("D", "C", "->"),
        ]
        assert pdag.latents == {"D"}

    def test_all_neighbors(self):
        pdag = PDAG(edge_list=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")])

        assert pdag.get_neighbors(node="A") == {"B", "C"}
        assert pdag.get_neighbors(node="B") == {"A", "D"}
        assert pdag.get_neighbors(node="C") == {"A", "D"}
        assert pdag.get_neighbors(node="D") == {"B", "C"}

    def test_get_children(self):
        pdag = PDAG(edge_list=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")])

        assert pdag.get_children("A") == {"C"}
        assert pdag.get_children("B") == set()
        assert pdag.get_children("C") == set()

    def test_get_parents(self):
        pdag = PDAG(edge_list=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")])

        assert pdag.get_parents("A") == set()
        assert pdag.get_parents("B") == set()
        assert pdag.get_parents("C") == {"A", "D"}

    def test_get_neighbors(self):
        pdag = PDAG(edge_list=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")])

        assert pdag.get_neighbors(node="A", edge_types="--") == {"B"}
        assert pdag.get_neighbors(node="B", edge_types="--") == {"A", "D"}
        assert pdag.get_neighbors(node="C", edge_types="--") == set()
        assert pdag.get_neighbors(node="D", edge_types="--") == {"B"}

    def test_chain_component(self):
        pdag = PDAG([("E", "F", "->"), ("A", "B", "--"), ("B", "C", "--"), ("D", "C", "--")])

        assert pdag.chain_component("A") == {"A", "B", "C", "D"}
        assert pdag.chain_component("C") == {"A", "B", "C", "D"}
        assert pdag.chain_component("E") == {"E"}
        assert pdag.chain_component("F") == {"F"}

    def test_has_semidirected_path(self):
        pdag = PDAG([("A", "B", "->"), ("C", "D", "->"), ("B", "C", "--")])

        assert pdag.has_semidirected_path("A", "D") is True
        assert pdag.has_semidirected_path("D", "A") is False
        assert pdag.has_semidirected_path("A", "D", blocked_nodes={"C"}) is False
        assert pdag.has_semidirected_path("A", "B", ignore_direct_edge=True) is False

    def test_acyclic(self):
        ebunch = [("A", "B", "->"), ("B", "C", "--")]
        PDAG(ebunch)

        with pytest.raises(ValueError):
            directed_cycle = [("A", "B", "->"), ("B", "C", "->"), ("C", "A", "->")]
            PDAG(directed_cycle)

    def test_to_cpdag(self):
        pdag = PDAG([("A", "B", "->"), ("B", "C", "--")])

        cpdag = pdag.to_cpdag()

        assert cpdag.directed_edges == set()
        assert cpdag.undirected_edges == {("A", "B"), ("B", "C")}

    def test_orient_undirected_edge(self):
        pdag = PDAG(edge_list=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")])

        mod_pdag = pdag.orient_undirected_edge("B", "A", inplace=False)
        assert sorted(mod_pdag.get_edges(data=True)) == [
            ("A", "C", "->"),
            ("B", "A", "->"),
            ("D", "B", "--"),
            ("D", "C", "->"),
        ]

        pdag.orient_undirected_edge("B", "A", inplace=True)
        assert sorted(pdag.get_edges(data=True)) == [
            ("A", "C", "->"),
            ("B", "A", "->"),
            ("D", "B", "--"),
            ("D", "C", "->"),
        ]

        with pytest.raises(ValueError):
            pdag.orient_undirected_edge("B", "A", inplace=True)

    def test_copy(self):
        ebunch = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag_mix = PDAG(ebunch)
        pdag_copy = pdag_mix.copy()
        assert set(pdag_copy.nodes()) == {"A", "B", "C", "D"}
        assert sorted(pdag_copy.get_edges(data=True)) == [
            ("A", "B", "--"),
            ("A", "C", "->"),
            ("D", "B", "--"),
            ("D", "C", "->"),
        ]
        assert pdag_copy.latents == set()

        ebunch = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag_latent = PDAG(
            ebunch,
            latents=["A", "D"],
        )
        pdag_copy = pdag_latent.copy()
        assert set(pdag_copy.nodes()) == {"A", "B", "C", "D"}
        assert sorted(pdag_copy.get_edges(data=True)) == [
            ("A", "B", "--"),
            ("A", "C", "->"),
            ("D", "B", "--"),
            ("D", "C", "->"),
        ]
        assert pdag_copy.latents == {"A", "D"}

        ebunch = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag_role = PDAG(
            ebunch,
            roles={"exposures": "A", "adjustment": "D", "outcomes": "C"},
        )
        pdag_copy = pdag_role.copy()
        assert set(pdag_copy.nodes()) == {"A", "B", "C", "D"}
        assert sorted(pdag_copy.get_edges(data=True)) == [
            ("A", "B", "--"),
            ("A", "C", "->"),
            ("D", "B", "--"),
            ("D", "C", "->"),
        ]
        assert pdag_copy.latents == set()
        assert pdag_copy.get_role("exposures") == ["A"]
        assert pdag_copy.get_role("adjustment") == ["D"]
        assert pdag_copy.get_role("outcomes") == ["C"]
        assert sorted(pdag_copy.get_roles()) == sorted(["adjustment", "exposures", "outcomes"])

        ebunch = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag_role_set = PDAG(
            ebunch,
            roles={"exposures": ("A", "D"), "outcomes": ("C")},
        )
        pdag_copy = pdag_role_set.copy()
        assert set(pdag_copy.nodes()) == {"A", "B", "C", "D"}
        assert sorted(pdag_copy.get_edges(data=True)) == [
            ("A", "B", "--"),
            ("A", "C", "->"),
            ("D", "B", "--"),
            ("D", "C", "->"),
        ]
        assert pdag_copy.latents == set()
        assert sorted(pdag_copy.get_role("exposures")) == sorted(["A", "D"])
        assert pdag_copy.get_role("outcomes") == ["C"]
        assert sorted(pdag_copy.get_roles()) == sorted(["exposures", "outcomes"])

        ebunch = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag_role_list = PDAG(
            ebunch,
            roles={"exposures": ["A", "D"], "outcomes": ["C"]},
        )
        pdag_copy = pdag_role_list.copy()
        assert set(pdag_copy.nodes()) == {"A", "B", "C", "D"}
        assert sorted(pdag_copy.get_edges(data=True)) == [
            ("A", "B", "--"),
            ("A", "C", "->"),
            ("D", "B", "--"),
            ("D", "C", "->"),
        ]
        assert pdag_copy.latents == set()
        assert sorted(pdag_copy.get_role("exposures")) == sorted(["A", "D"])
        assert pdag_copy.get_role("outcomes") == ["C"]
        assert sorted(pdag_copy.get_roles()) == sorted(["exposures", "outcomes"])

    def test_pdag_to_dag(self):
        # PDAG no: 1  Possibility of creating a v-structure
        pdag = PDAG(edge_list=[("A", "B", "->"), ("C", "B", "->"), ("C", "D", "--"), ("D", "A", "--")])
        dag = pdag.to_dag()
        assert ("A", "B") in dag.edges()
        assert ("C", "B") in dag.edges()
        assert not ((("A", "D") in dag.edges()) and (("C", "D") in dag.edges()))
        assert len(dag.edges()) == 4

        # With latents
        pdag = PDAG(
            edge_list=[("A", "B", "->"), ("C", "B", "->"), ("C", "D", "--"), ("D", "A", "--")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        assert ("A", "B") in dag.edges()
        assert ("C", "B") in dag.edges()
        assert not ((("A", "D") in dag.edges()) and (("C", "D") in dag.edges()))
        assert dag.latents == {"A"}
        assert len(dag.edges()) == 4

        # PDAG no: 2  No possibility of creation of v-structure.
        pdag = PDAG(edge_list=[("B", "C", "->"), ("A", "C", "->"), ("A", "D", "--")])
        dag = pdag.to_dag()
        assert ("B", "C") in dag.edges()
        assert ("A", "C") in dag.edges()
        assert (("A", "D") in dag.edges()) or (("D", "A") in dag.edges())

        # With latents
        pdag = PDAG(
            edge_list=[("B", "C", "->"), ("A", "C", "->"), ("A", "D", "--")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        assert ("B", "C") in dag.edges()
        assert ("A", "C") in dag.edges()
        assert (("A", "D") in dag.edges()) or (("D", "A") in dag.edges())
        assert dag.latents == {"A"}

        # PDAG no: 3  Already existing v-structure, possibility to add another
        pdag = PDAG(edge_list=[("B", "C", "->"), ("A", "C", "->"), ("C", "D", "--")])
        dag = pdag.to_dag()
        expected_edges = {("B", "C"), ("C", "D"), ("A", "C")}
        assert expected_edges == set(dag.edges())

        # With latents
        pdag = PDAG(
            edge_list=[("B", "C", "->"), ("A", "C", "->"), ("C", "D", "--")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        expected_edges = {("B", "C"), ("C", "D"), ("A", "C")}
        assert expected_edges == set(dag.edges())
        assert dag.latents == {"A"}

        ebunch = [
            (1, 4, "--"),
            (5, 0, "--"),
            (0, 2, "->"),
            (1, 2, "->"),
            (3, 1, "->"),
            (3, 2, "->"),
            (3, 4, "->"),
            (4, 2, "->"),
            (5, 1, "->"),
            (5, 2, "->"),
            (5, 4, "->"),
        ]
        pdag = PDAG(edge_list=ebunch)
        dag = pdag.to_dag()
        dag_actual = {
            (0, 2),
            (1, 2),
            (3, 1),
            (3, 2),
            (3, 4),
            (4, 1),
            (4, 2),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 4),
        }
        assert set(dag.edges()) == dag_actual

    def test_pdag_to_cpdag(self):
        pdag = PDAG(edge_list=[("A", "B", "->"), ("B", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        assert set(cpdag.get_edges(data=True)) == {
            ("A", "B", "->"),
            ("B", "C", "->"),
        }

        pdag = PDAG(edge_list=[("A", "B", "->"), ("B", "C", "--"), ("C", "D", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        assert set(cpdag.get_edges(data=True)) == {
            ("A", "B", "->"),
            ("B", "C", "->"),
            ("C", "D", "->"),
        }

        pdag = PDAG(edge_list=[("A", "B", "->"), ("D", "C", "->"), ("B", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        assert set(cpdag.get_edges(data=True)) == {
            ("A", "B", "->"),
            ("B", "C", "--"),
            ("D", "C", "->"),
        }

        pdag = PDAG(edge_list=[("A", "B", "->"), ("D", "C", "->"), ("D", "B", "->"), ("B", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        assert set(cpdag.get_edges(data=True)) == {
            ("A", "B", "->"),
            ("B", "C", "->"),
            ("D", "B", "->"),
            ("D", "C", "->"),
        }

        pdag = PDAG(edge_list=[("A", "B", "->"), ("B", "C", "->"), ("A", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        assert set(cpdag.get_edges(data=True)) == {
            ("A", "B", "->"),
            ("A", "C", "->"),
            ("B", "C", "->"),
        }

        pdag = PDAG(edge_list=[("A", "B", "->"), ("B", "C", "->"), ("D", "C", "->"), ("A", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        assert set(cpdag.get_edges(data=True)) == {
            ("A", "B", "->"),
            ("A", "C", "->"),
            ("B", "C", "->"),
            ("D", "C", "->"),
        }

        # Examples taken from Perković 2017.
        pdag = PDAG(edge_list=[("V1", "X", "->"), ("X", "V2", "--"), ("V2", "Y", "--"), ("X", "Y", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        assert set(cpdag.get_edges(data=True)) == {
            ("V1", "X", "->"),
            ("V2", "Y", "--"),
            ("X", "V2", "->"),
            ("X", "Y", "->"),
        }

        pdag = PDAG(edge_list=[("Y", "X", "->"), ("V1", "X", "--"), ("X", "V2", "--"), ("V2", "Y", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        assert set(cpdag.get_edges(data=True)) == {
            ("X", "V1", "->"),
            ("X", "V2", "--"),
            ("Y", "X", "->"),
            ("Y", "V2", "--"),
        }

        # Examples from Bang 2024
        pdag = PDAG(edge_list=[("B", "D", "->"), ("C", "D", "->"), ("A", "D", "--"), ("A", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True, debug=True)
        assert set(cpdag.get_edges(data=True)) == {
            ("B", "D", "->"),
            ("C", "A", "->"),
            ("C", "D", "->"),
            ("D", "A", "->"),
        }

        pdag = PDAG(
            edge_list=[("A", "B", "->"), ("C", "B", "->"), ("D", "B", "--"), ("D", "A", "--"), ("D", "C", "--")]
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        assert set(cpdag.get_edges(data=True)) == {
            ("A", "B", "->"),
            ("A", "D", "--"),
            ("C", "B", "->"),
            ("C", "D", "--"),
            ("D", "B", "->"),
        }

        ebunch = [("A", "C", "--"), ("B", "C", "--"), ("D", "C", "--"), ("B", "D", "->"), ("D", "A", "->")]

        pdag = PDAG(edge_list=ebunch)
        mpdag = pdag.apply_meeks_rules(apply_r4=True)
        assert set(mpdag.get_edges(data=True)) == {
            ("C", "B", "--"),
            ("B", "D", "->"),
            ("C", "A", "->"),
            ("C", "D", "--"),
            ("D", "A", "->"),
        }

        pdag = PDAG(edge_list=ebunch)
        pdag = pdag.apply_meeks_rules()
        assert set(pdag.get_edges(data=True)) == {
            ("A", "C", "--"),
            ("C", "B", "--"),
            ("B", "D", "->"),
            ("C", "D", "--"),
            ("D", "A", "->"),
        }

        pdag_inp = PDAG(edge_list=ebunch)
        pdag_inp.apply_meeks_rules(inplace=True)
        assert set(pdag_inp.get_edges(data=True)) == {
            ("A", "C", "--"),
            ("C", "B", "--"),
            ("B", "D", "->"),
            ("C", "D", "--"),
            ("D", "A", "->"),
        }

    def test_meeks_rules_skip_cycle_creating_orientation(self):
        # On an inconsistent PDAG a Meek orientation can close a directed cycle. orient_undirected_edge
        # now refuses that (raises); R3/R4 must skip such orientations (as R1 already does), not crash.
        # R3: x--{w,y,z}, y->w<-z would orient x->w, but w->A->B->x already exists.
        out = PDAG(
            edge_list=[
                ("x", "w", "--"),
                ("x", "y", "--"),
                ("x", "z", "--"),
                ("y", "w", "->"),
                ("z", "w", "->"),
                ("w", "A", "->"),
                ("A", "B", "->"),
                ("B", "x", "->"),
            ]
        ).apply_meeks_rules()
        assert ("x", "w") not in out.directed_edges
        # R4: d->c->b, a--{b,c,d}, b not adj d would orient a->b, but b->P->Q->a already exists.
        out4 = PDAG(
            edge_list=[
                ("d", "c", "->"),
                ("c", "b", "->"),
                ("a", "b", "--"),
                ("a", "c", "--"),
                ("a", "d", "--"),
                ("b", "P", "->"),
                ("P", "Q", "->"),
                ("Q", "a", "->"),
            ]
        ).apply_meeks_rules(apply_r4=True)
        assert ("a", "b") not in out4.directed_edges

    def test_meeks_rule3_requires_nonadjacent_parents(self):
        # R3 compels x->w only when w's two parents are non-adjacent. Here c--d are adjacent, so
        # a->b is not compelled and must stay undirected.
        out = PDAG(
            edge_list=[
                ("a", "b", "--"),
                ("a", "c", "--"),
                ("a", "d", "--"),
                ("c", "b", "->"),
                ("d", "b", "->"),
                ("c", "d", "--"),
            ]
        ).apply_meeks_rules()
        assert ("a", "b") not in out.directed_edges
        assert out.directed_edges == {("c", "b"), ("d", "b")}

    def test_to_dag_preserves_skeleton_on_nonextendable_pdag(self):
        # A chordless 4-cycle has no faithful extension, so to_dag's fallback orients edges
        # arbitrarily; it must still keep every edge (never silently drop a cycle-closing one).
        pdag = PDAG(edge_list=[("A", "B", "--"), ("B", "C", "--"), ("C", "D", "--"), ("D", "A", "--")])
        dag = pdag.to_dag()
        assert {frozenset(e) for e in dag.edges()} == {
            frozenset(("A", "B")),
            frozenset(("B", "C")),
            frozenset(("C", "D")),
            frozenset(("D", "A")),
        }
        assert len(dag.edges()) == 4  # acyclic orientation, nothing dropped

    def test_pdag_equality(self):
        """
        Test the `__eq__` method
        which compares both graph structure and variable-role mappings to allow comparison of two models.
        """
        pdag = PDAG(
            edge_list=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )

        # Case1: When the models are the same
        other1 = PDAG(
            edge_list=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case2: When the models differ
        other2 = DAG(
            ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "C", "->")],
            latents=["B"],
            roles={"exposures": "A", "adjustment": "D", "outcomes": "C"},
        )
        # Case3: When the directed edges differ between models
        other3 = PDAG(
            edge_list=[("A", "C", "->"), ("D", "C", "->"), ("E", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case4: When the directed edges differ between models
        other4 = PDAG(
            edge_list=[("A", "E", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case5: When the undirected edges differ between models
        other5 = PDAG(
            edge_list=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "E", "--")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case6: When the latents variables differ between models
        other6 = PDAG(
            edge_list=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["D"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case7: When the roles variables differ between models
        other7 = PDAG(
            edge_list=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["B"],
            roles={"exposures": ("A"), "adjustment": "D", "outcomes": ["C"]},
        )

        assert pdag.__eq__(other1) is True
        assert pdag.__eq__(other2) is False
        assert pdag.__eq__(other3) is False
        assert pdag.__eq__(other4) is False
        assert pdag.__eq__(other5) is False
        assert pdag.__eq__(other6) is False
        assert pdag.__eq__(other7) is False

    def test_latents_with_role(self):
        pdag1 = PDAG(
            edge_list=[
                ("X", "Y", "->"),
                ("A", "B", "--"),
                ("B", "C", "--"),
                ("C", "D", "--"),
                ("D", "E", "--"),
                ("E", "F", "--"),
            ],
            latents=["A"],
            roles={"exposures": "X", "outcomes": "Y", "latents": "B"},
        )
        pdag1.with_role(role="latents", variables="C", inplace=True)
        pdag1.with_role(role="latents", variables=["D", "E"], inplace=True)
        pdag1 = pdag1.with_role(role="latents", variables="F", inplace=False)

        assert pdag1.latents == {"A", "B", "C", "D", "E", "F"}
        assert set(pdag1.get_role("latents")) == {"A", "B", "C", "D", "E", "F"}

        with pytest.raises(ValueError, match="Variable 'G' not found in the graph."):
            pdag1.with_role(role="latents", variables="G", inplace=True)

    def test_latents_without_role(self):
        pdag1 = PDAG(
            edge_list=[
                ("X", "Y", "->"),
                ("A", "B", "--"),
                ("B", "C", "--"),
                ("C", "D", "--"),
                ("D", "E", "--"),
                ("E", "F", "--"),
            ],
            latents=["A", "B", "C"],
            roles={"exposures": "X", "outcomes": "Y", "latents": ("D", "E", "F")},
        )

        pdag1.without_role(role="latents", variables="A", inplace=True)
        pdag1.without_role(role="latents", variables=["B", "C"], inplace=True)
        pdag1 = pdag1.without_role(role="latents", variables="D", inplace=False)
        pdag1 = pdag1.without_role(role="latents", variables=["E", "F"], inplace=False)

        assert pdag1.latents == set()
        assert set(pdag1.get_role("latents")) == set()

    def test_is_clique(self):
        """Test code for `is_clique()` method"""
        pdag = PDAG([("A", "B", "--"), ("B", "C", "--"), ("A", "C", "--"), ("C", "D", "--")])
        assert pdag.is_clique({"A", "B", "C"}) is True
        assert pdag.is_clique({"A", "B", "C", "D"}) is False  # A-D not adjacent
        assert pdag.is_clique({"A"}) is True
