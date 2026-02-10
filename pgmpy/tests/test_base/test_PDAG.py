import unittest

from pgmpy.base import DAG, PDAG


class TestPDAG(unittest.TestCase):
    def setUp(self):
        self.pdag_mix = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
        )
        self.pdag_dir = PDAG(
            directed_ebunch=[("A", "B"), ("D", "B"), ("A", "C"), ("D", "C")]
        )
        self.pdag_undir = PDAG(
            undirected_ebunch=[("A", "C"), ("D", "C"), ("B", "A"), ("B", "D")]
        )
        self.pdag_latent = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["A", "D"],
        )
        self.pdag_role = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            roles={"exposures": "A", "adjustment": "D", "outcomes": "C"},
        )
        self.pdag_role_set = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            roles={"exposures": ("A", "D"), "outcomes": ("C")},
        )
        self.pdag_role_list = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            roles={"exposures": ["A", "D"], "outcomes": ["C"]},
        )

    def test_init_normal(self):
        # Mix directed and undirected
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag.edges()), expected_edges)
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set(directed_edges))
        self.assertEqual(pdag.undirected_edges, set(undirected_edges))

        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(
            directed_ebunch=directed_edges,
            undirected_ebunch=undirected_edges,
            latents=["A", "C"],
        )
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag.edges()), expected_edges)
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set(directed_edges))
        self.assertEqual(pdag.undirected_edges, set(undirected_edges))
        self.assertEqual(pdag.latents, set(["A", "C"]))

        # Only undirected
        undirected_edges = [("A", "C"), ("D", "C"), ("B", "A"), ("B", "D")]
        pdag = PDAG(undirected_ebunch=undirected_edges)
        expected_edges = {
            ("A", "C"),
            ("C", "A"),
            ("D", "C"),
            ("C", "D"),
            ("B", "A"),
            ("A", "B"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag.edges()), expected_edges)
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set())
        self.assertEqual(pdag.undirected_edges, set(undirected_edges))

        undirected_edges = [("A", "C"), ("D", "C"), ("B", "A"), ("B", "D")]
        pdag = PDAG(undirected_ebunch=undirected_edges, latents=["A", "D"])
        expected_edges = {
            ("A", "C"),
            ("C", "A"),
            ("D", "C"),
            ("C", "D"),
            ("B", "A"),
            ("A", "B"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag.edges()), expected_edges)
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set())
        self.assertEqual(pdag.undirected_edges, set(undirected_edges))
        self.assertEqual(pdag.latents, set(["A", "D"]))

        # Only directed
        directed_edges = [("A", "B"), ("D", "B"), ("A", "C"), ("D", "C")]
        pdag = PDAG(directed_ebunch=directed_edges)
        self.assertEqual(set(pdag.edges()), set(directed_edges))
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set(directed_edges))
        self.assertEqual(pdag.undirected_edges, set())

        directed_edges = [("A", "B"), ("D", "B"), ("A", "C"), ("D", "C")]
        pdag = PDAG(directed_ebunch=directed_edges, latents=["D"])
        self.assertEqual(set(pdag.edges()), set(directed_edges))
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set(directed_edges))
        self.assertEqual(pdag.undirected_edges, set())
        self.assertEqual(pdag.latents, set(["D"]))

    def test_all_neighrors(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertEqual(pdag.all_neighbors(node="A"), {"B", "C"})
        self.assertEqual(pdag.all_neighbors(node="B"), {"A", "D"})
        self.assertEqual(pdag.all_neighbors(node="C"), {"A", "D"})
        self.assertEqual(pdag.all_neighbors(node="D"), {"B", "C"})

    def test_directed_children(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertEqual(pdag.directed_children(node="A"), {"C"})
        self.assertEqual(pdag.directed_children(node="B"), set())
        self.assertEqual(pdag.directed_children(node="C"), set())

    def test_directed_parents(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertEqual(pdag.directed_parents(node="A"), set())
        self.assertEqual(pdag.directed_parents(node="B"), set())
        self.assertEqual(pdag.directed_parents(node="C"), {"A", "D"})

    def test_has_directed_edge(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertTrue(pdag.has_directed_edge("A", "C"))
        self.assertTrue(pdag.has_directed_edge("D", "C"))
        self.assertFalse(pdag.has_directed_edge("A", "B"))
        self.assertFalse(pdag.has_directed_edge("B", "A"))

    def test_has_undirected_edge(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertFalse(pdag.has_undirected_edge("A", "C"))
        self.assertFalse(pdag.has_undirected_edge("D", "C"))
        self.assertTrue(pdag.has_undirected_edge("A", "B"))
        self.assertTrue(pdag.has_undirected_edge("B", "A"))

    def test_undirected_neighbors(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertEqual(pdag.undirected_neighbors(node="A"), {"B"})
        self.assertEqual(pdag.undirected_neighbors(node="B"), {"A", "D"})
        self.assertEqual(pdag.undirected_neighbors(node="C"), set())
        self.assertEqual(pdag.undirected_neighbors(node="D"), {"B"})

    def test_orient_undirected_edge(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        mod_pdag = pdag.orient_undirected_edge("B", "A", inplace=False)
        self.assertEqual(
            set(mod_pdag.edges()),
            {("A", "C"), ("D", "C"), ("B", "A"), ("B", "D"), ("D", "B")},
        )
        self.assertEqual(mod_pdag.undirected_edges, {("B", "D")})
        self.assertEqual(mod_pdag.directed_edges, {("A", "C"), ("D", "C"), ("B", "A")})

        pdag.orient_undirected_edge("B", "A", inplace=True)
        self.assertEqual(
            set(pdag.edges()),
            {("A", "C"), ("D", "C"), ("B", "A"), ("B", "D"), ("D", "B")},
        )
        self.assertEqual(pdag.undirected_edges, {("B", "D")})
        self.assertEqual(pdag.directed_edges, {("A", "C"), ("D", "C"), ("B", "A")})

        self.assertRaises(
            ValueError, pdag.orient_undirected_edge, "B", "A", inplace=True
        )

    def test_copy(self):
        pdag_copy = self.pdag_mix.copy()
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag_copy.edges()), expected_edges)
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag_copy.directed_edges, set([("A", "C"), ("D", "C")]))
        self.assertEqual(pdag_copy.undirected_edges, set([("B", "A"), ("B", "D")]))
        self.assertEqual(pdag_copy.latents, set())

        pdag_copy = self.pdag_latent.copy()
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag_copy.edges()), expected_edges)
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag_copy.directed_edges, set([("A", "C"), ("D", "C")]))
        self.assertEqual(pdag_copy.undirected_edges, set([("B", "A"), ("B", "D")]))
        self.assertEqual(pdag_copy.latents, set(["A", "D"]))

        pdag_copy = self.pdag_role.copy()
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag_copy.edges()), expected_edges)
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag_copy.directed_edges, set([("A", "C"), ("D", "C")]))
        self.assertEqual(pdag_copy.undirected_edges, set([("B", "A"), ("B", "D")]))
        self.assertEqual(pdag_copy.latents, set())
        self.assertEqual(pdag_copy.get_role("exposures"), ["A"])
        self.assertEqual(pdag_copy.get_role("adjustment"), ["D"])
        self.assertEqual(pdag_copy.get_role("outcomes"), ["C"])
        self.assertEqual(
            sorted(pdag_copy.get_roles()),
            sorted(["adjustment", "exposures", "outcomes"]),
        )

        pdag_copy = self.pdag_role_set.copy()
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag_copy.edges()), expected_edges)
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag_copy.directed_edges, set([("A", "C"), ("D", "C")]))
        self.assertEqual(pdag_copy.undirected_edges, set([("B", "A"), ("B", "D")]))
        self.assertEqual(pdag_copy.latents, set())
        self.assertEqual(sorted(pdag_copy.get_role("exposures")), sorted(["A", "D"]))
        self.assertEqual(pdag_copy.get_role("outcomes"), ["C"])
        self.assertEqual(
            sorted(pdag_copy.get_roles()), sorted(["exposures", "outcomes"])
        )

        pdag_copy = self.pdag_role_list.copy()
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag_copy.edges()), expected_edges)
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag_copy.directed_edges, set([("A", "C"), ("D", "C")]))
        self.assertEqual(pdag_copy.undirected_edges, set([("B", "A"), ("B", "D")]))
        self.assertEqual(pdag_copy.latents, set())
        self.assertEqual(sorted(pdag_copy.get_role("exposures")), sorted(["A", "D"]))
        self.assertEqual(pdag_copy.get_role("outcomes"), ["C"])
        self.assertEqual(
            sorted(pdag_copy.get_roles()), sorted(["exposures", "outcomes"])
        )

    def test_pdag_to_dag(self):
        # PDAG no: 1  Possibility of creating a v-structure
        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("C", "B")],
            undirected_ebunch=[("C", "D"), ("D", "A")],
        )
        dag = pdag.to_dag()
        self.assertTrue(("A", "B") in dag.edges())
        self.assertTrue(("C", "B") in dag.edges())
        self.assertFalse((("A", "D") in dag.edges()) and (("C", "D") in dag.edges()))
        self.assertTrue(len(dag.edges()) == 4)

        # With latents
        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("C", "B")],
            undirected_ebunch=[("C", "D"), ("D", "A")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        self.assertTrue(("A", "B") in dag.edges())
        self.assertTrue(("C", "B") in dag.edges())
        self.assertFalse((("A", "D") in dag.edges()) and (("C", "D") in dag.edges()))
        self.assertEqual(dag.latents, set(["A"]))
        self.assertTrue(len(dag.edges()) == 4)

        # PDAG no: 2  No possibility of creation of v-structure.
        pdag = PDAG(
            directed_ebunch=[("B", "C"), ("A", "C")], undirected_ebunch=[("A", "D")]
        )
        dag = pdag.to_dag()
        self.assertTrue(("B", "C") in dag.edges())
        self.assertTrue(("A", "C") in dag.edges())
        self.assertTrue((("A", "D") in dag.edges()) or (("D", "A") in dag.edges()))

        # With latents
        pdag = PDAG(
            directed_ebunch=[("B", "C"), ("A", "C")],
            undirected_ebunch=[("A", "D")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        self.assertTrue(("B", "C") in dag.edges())
        self.assertTrue(("A", "C") in dag.edges())
        self.assertTrue((("A", "D") in dag.edges()) or (("D", "A") in dag.edges()))
        self.assertEqual(dag.latents, set(["A"]))

        # PDAG no: 3  Already existing v-structure, possibility to add another
        pdag = PDAG(
            directed_ebunch=[("B", "C"), ("A", "C")], undirected_ebunch=[("C", "D")]
        )
        dag = pdag.to_dag()
        expected_edges = {("B", "C"), ("C", "D"), ("A", "C")}
        self.assertEqual(expected_edges, set(dag.edges()))

        # With latents
        pdag = PDAG(
            directed_ebunch=[("B", "C"), ("A", "C")],
            undirected_ebunch=[("C", "D")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        expected_edges = {("B", "C"), ("C", "D"), ("A", "C")}
        self.assertEqual(expected_edges, set(dag.edges()))
        self.assertEqual(dag.latents, set(["A"]))

        undirected_edges = [(1, 4), (5, 0)]
        directed_edges = [
            (0, 2),
            (1, 2),
            (3, 1),
            (3, 2),
            (3, 4),
            (4, 2),
            (5, 1),
            (5, 2),
            (5, 4),
        ]
        pdag = PDAG(undirected_ebunch=undirected_edges, directed_ebunch=directed_edges)
        dag = pdag.to_dag()
        dag_actual = set(
            [
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
            ]
        )
        self.assertSetEqual(set(dag.edges), dag_actual)

    def test_pdag_to_cpdag(self):
        pdag = PDAG(directed_ebunch=[("A", "B")], undirected_ebunch=[("B", "C")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C")})

        pdag = PDAG(
            directed_ebunch=[("A", "B")], undirected_ebunch=[("B", "C"), ("C", "D")]
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C"), ("C", "D")})

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("D", "C")], undirected_ebunch=[("B", "C")]
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()), {("A", "B"), ("D", "C"), ("B", "C"), ("C", "B")}
        )

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("D", "C"), ("D", "B")],
            undirected_ebunch=[("B", "C")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()), {("A", "B"), ("D", "C"), ("D", "B"), ("B", "C")}
        )

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("B", "C")], undirected_ebunch=[("A", "C")]
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C"), ("A", "C")})

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("B", "C"), ("D", "C")],
            undirected_ebunch=[("A", "C")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()), {("A", "B"), ("B", "C"), ("A", "C"), ("D", "C")}
        )

        # Examples taken from Perkovi\`c 2017.
        pdag = PDAG(
            directed_ebunch=[("V1", "X")],
            undirected_ebunch=[("X", "V2"), ("V2", "Y"), ("X", "Y")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertEqual(
            set(cpdag.edges()),
            {("V1", "X"), ("X", "V2"), ("X", "Y"), ("V2", "Y"), ("Y", "V2")},
        )

        pdag = PDAG(
            directed_ebunch=[("Y", "X")],
            undirected_ebunch=[("V1", "X"), ("X", "V2"), ("V2", "Y")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertEqual(
            set(cpdag.edges()),
            {
                ("X", "V1"),
                ("Y", "X"),
                ("X", "V2"),
                ("V2", "X"),
                ("V2", "Y"),
                ("Y", "V2"),
            },
        )

        # Examples from Bang 2024
        pdag = PDAG(
            directed_ebunch=[("B", "D"), ("C", "D")],
            undirected_ebunch=[("A", "D"), ("A", "C")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True, debug=True)
        self.assertEqual(
            set(cpdag.edges()), {("B", "D"), ("D", "A"), ("C", "A"), ("C", "D")}
        )

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("C", "B")],
            undirected_ebunch=[("D", "B"), ("D", "A"), ("D", "C")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()),
            {
                ("A", "B"),
                ("C", "B"),
                ("D", "B"),
                ("D", "A"),
                ("A", "D"),
                ("D", "C"),
                ("C", "D"),
            },
        )

        undirected_edges = [("A", "C"), ("B", "C"), ("D", "C")]
        directed_edges = [("B", "D"), ("D", "A")]

        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
        mpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(mpdag.edges()),
            set(
                [
                    ("C", "A"),
                    ("C", "B"),
                    ("B", "C"),
                    ("B", "D"),
                    ("D", "A"),
                    ("D", "C"),
                    ("C", "D"),
                ]
            ),
        )

        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
        pdag = pdag.apply_meeks_rules()
        self.assertSetEqual(
            set(pdag.edges()),
            set(
                [
                    ("A", "C"),
                    ("C", "A"),
                    ("C", "B"),
                    ("B", "C"),
                    ("B", "D"),
                    ("D", "A"),
                    ("D", "C"),
                    ("C", "D"),
                ]
            ),
        )

        pdag_inp = PDAG(
            directed_ebunch=directed_edges, undirected_ebunch=undirected_edges
        )
        pdag_inp.apply_meeks_rules(inplace=True)
        self.assertSetEqual(
            set(pdag_inp.edges()),
            set(
                [
                    ("A", "C"),
                    ("C", "A"),
                    ("C", "B"),
                    ("B", "C"),
                    ("B", "D"),
                    ("D", "A"),
                    ("D", "C"),
                    ("C", "D"),
                ]
            ),
        )

    def test_pdag_equality(self):
        """
        Test the `__eq__` method
        which compares both graph structure and variable-role mappings to allow comparison of two models.
        """
        pdag = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )

        # Case1: When the models are the same
        other1 = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case2: When the models differ
        other2 = DAG(
            ebunch=[("A", "C"), ("D", "C"), ("B", "C")],
            latents=["B"],
            roles={"exposures": "A", "adjustment": "D", "outcomes": "C"},
        )
        # Case3: When the directed_ebunch variables differ between models
        other3 = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C"), ("E", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case4: When the directed_ebunch variables differ between models
        other4 = PDAG(
            directed_ebunch=[("A", "E"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case5: When the undirected_ebunch variables differ between models
        other5 = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "E")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case6: When the latents variables differ between models
        other6 = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["D"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case7: When the roles variables differ between models
        other7 = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["B"],
            roles={"exposures": ("A"), "adjustment": "D", "outcomes": ["C"]},
        )

        self.assertEqual(pdag.__eq__(other1), True)
        self.assertEqual(pdag.__eq__(other2), False)
        self.assertEqual(pdag.__eq__(other3), False)
        self.assertEqual(pdag.__eq__(other4), False)
        self.assertEqual(pdag.__eq__(other5), False)
        self.assertEqual(pdag.__eq__(other6), False)
        self.assertEqual(pdag.__eq__(other7), False)

    def test_latents_with_role(self):
        self.pdag1 = PDAG(
            directed_ebunch=[("X", "Y")],
            undirected_ebunch=[
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("D", "E"),
                ("E", "F"),
            ],
            latents=["A"],
            roles={"exposures": "X", "outcomes": "Y", "latents": "B"},
        )
        self.pdag1.with_role(role="latents", variables="C", inplace=True)
        self.pdag1.with_role(role="latents", variables=["D", "E"], inplace=True)
        self.pdag1 = self.pdag1.with_role(role="latents", variables="F", inplace=False)

        self.assertEqual(self.pdag1.latents, {"A", "B", "C", "D", "E", "F"})
        self.assertEqual(
            set(self.pdag1.get_role("latents")), {"A", "B", "C", "D", "E", "F"}
        )

        with self.assertRaisesRegex(ValueError, "Variable 'G' not found in the graph."):
            self.pdag1.with_role(role="latents", variables="G", inplace=True)

    def test_latnets_without_role(self):
        self.pdag1 = PDAG(
            directed_ebunch=[("X", "Y")],
            undirected_ebunch=[
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("D", "E"),
                ("E", "F"),
            ],
            latents=["A", "B", "C"],
            roles={"exposures": "X", "outcomes": "Y", "latents": ("D", "E", "F")},
        )

        self.pdag1.without_role(role="latents", variables="A", inplace=True)
        self.pdag1.without_role(role="latents", variables=["B", "C"], inplace=True)
        self.pdag1 = self.pdag1.without_role(
            role="latents", variables="D", inplace=False
        )
        self.pdag1 = self.pdag1.without_role(
            role="latents", variables=["E", "F"], inplace=False
        )

        self.assertEqual(self.pdag1.latents, set())
        self.assertEqual(set(self.pdag1.get_role("latents")), set())
