import unittest
from pgmpy.base.causal_graph import CausalGraph
from pgmpy.base.DAG import DAG


class TestCausalGraph(unittest.TestCase):
    def setUp(self):
        self.edges = [("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")]
        self.cg = CausalGraph(graph=self.edges, exposure="X", outcome="Y")

    def test_init_with_edges_and_roles(self):
        cg = CausalGraph(graph=self.edges, exposure="X", outcome="Y")
        self.assertEqual(cg.get_role("exposure"), {"X"})
        self.assertEqual(cg.get_role("outcome"), {"Y"})
        self.assertEqual(set(cg.nodes()), {"U", "X", "M", "Y"})
        self.assertEqual(set(cg.edges()), set(self.edges))

    def test_init_with_DAG(self):
        dag = DAG(self.edges)
        cg = CausalGraph(graph=dag, exposure="X", outcome="Y")
        self.assertEqual(set(cg.nodes()), set(dag.nodes()))
        self.assertEqual(set(cg.edges()), set(dag.edges()))

    def test_role_aliases(self):
        cg = CausalGraph(graph=self.edges, treatment="X", target="Y")
        self.assertEqual(cg.get_role("exposure"), {"X"})
        self.assertEqual(cg.get_role("outcome"), {"Y"})
        self.assertTrue(cg.has_role("exposure"))
        self.assertTrue(cg.has_role("outcome"))

    def test_roles_dict_and_kwargs(self):
        cg = CausalGraph(
            graph=self.edges, roles={"adjustment": {"U", "M"}}, exposure="X"
        )
        self.assertEqual(cg.get_role("adjustment"), {"U", "M"})
        self.assertEqual(cg.get_role("exposure"), {"X"})

    def test_with_role_and_without_role(self):
        cg2 = self.cg.with_role("adjustment", {"U", "M"})
        self.assertEqual(cg2.get_role("adjustment"), {"U", "M"})
        cg3 = cg2.without_role("adjustment")
        self.assertFalse(cg3.has_role("adjustment"))

    def test_get_roles(self):
        cg = CausalGraph(
            graph=self.edges, exposure="X", outcome="Y", adjustment={"U", "M"}
        )
        roles = cg.get_roles()
        self.assertEqual(roles["exposure"], {"X"})
        self.assertEqual(roles["outcome"], {"Y"})
        self.assertEqual(roles["adjustment"], {"U", "M"})

    def test_validate_roles_success(self):
        cg = CausalGraph(
            graph=self.edges, exposure="X", outcome="Y", adjustment={"U", "M"}
        )
        self.assertTrue(cg.validate_roles())

    def test_validate_roles_invalid(self):
        cg = CausalGraph(
            graph=self.edges, exposure="X", outcome="Y", adjustment={"U", "Z"}
        )
        with self.assertRaises(ValueError):
            cg.validate_roles()

    def test_with_role_invalid_variable(self):
        with self.assertRaises(ValueError):
            self.cg.with_role("adjustment", {"Z"})

    def test_copy_and_equality(self):
        cg2 = self.cg.copy()
        self.assertEqual(self.cg, cg2)
        cg3 = self.cg.with_role("adjustment", {"U"})
        self.assertNotEqual(self.cg, cg3)

    def test_hash(self):
        cg2 = self.cg.copy()
        self.assertEqual(hash(self.cg), hash(cg2))
        cg3 = self.cg.with_role("adjustment", {"U"})
        self.assertNotEqual(hash(self.cg), hash(cg3))

    def test_to_dag(self):
        dag = self.cg.to_dag()
        self.assertIsInstance(dag, DAG)
        self.assertEqual(set(dag.nodes()), set(self.cg.nodes()))
        self.assertEqual(set(dag.edges()), set(self.cg.edges()))

    def test_is_valid_causal_structure(self):
        cg = CausalGraph(graph=self.edges, exposure="X", outcome="Y")
        self.assertTrue(cg.is_valid_causal_structure())
        cg2 = CausalGraph(graph=self.edges, exposure=["X", "M"], outcome="Y")
        with self.assertRaises(ValueError):
            cg2.is_valid_causal_structure()

    def test_with_nodes_and_with_edges(self):
        cg2 = self.cg.with_nodes(["Z"])
        self.assertIn("Z", cg2.nodes())
        cg3 = self.cg.with_edges([("X", "Z")])
        self.assertIn(("X", "Z"), cg3.edges())

    def test_without_nodes_and_without_edges(self):
        cg2 = self.cg.without_nodes(["U"])
        self.assertNotIn("U", cg2.nodes())
        cg3 = self.cg.without_edges([("U", "X")])
        self.assertNotIn(("U", "X"), cg3.edges())

    def test_str_and_repr(self):
        s = str(self.cg)
        r = repr(self.cg)
        self.assertIn("CausalGraph", s)
        self.assertEqual(s, r)

    def test_getattr_forwarding(self):
        self.assertEqual(set(self.cg.get_parents("X")), {"U"})

    def tearDown(self):
        pass
