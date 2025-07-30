import pytest

from pgmpy.base.causal_graph import CausalGraph
from pgmpy.base.DAG import DAG


@pytest.fixture
def cg():
    edges = [("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")]
    roles = {"X": "exposure", "Y": "outcome"}
    return CausalGraph(ebunch=edges, roles=roles)


@pytest.fixture
def cg2():
    cg2 = CausalGraph(
        ebunch=[("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")]
        roles={"U": "adjustment", "M": "adjustment", "X": "exposure"},
    )
    return cg2


class TestCausalGraph:

    def test_init_with_edges_and_roles(self, cg):
        assert cg.get_role("exposure") == ["X"]
        assert cg.get_role("outcome") == ["Y"]
        assert set(cg.nodes()) == {"U", "X", "M", "Y"}

    def test_roles_dict_and_kwargs(self, cg2):
        cg = cg2
        assert set(cg.get_role("adjustment")) == {"U", "M"}
        assert set(cg.get_role("exposure")) == {"X"}

    def test_with_role_and_without_role(self, cg):
        cg2 = cg.with_role("adjustment", {"U", "M"})
        assert set(cg2.get_role("adjustment")) == {"U", "M"}
        cg3 = cg2.without_role("adjustment")
        assert cg3.has_role("adjustment")

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

    def test_is_valid_causal_structure(self, cg):
        assert cg.is_valid_causal_structure()
        cg2 = CausalGraph(
            ebunch=[("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")]
            roles={"Y": "outcome", "M": "exposure", "X": "exposure"},
        )
        with pytest.raises(ValueError):
            cg2.is_valid_causal_structure()

    def test_with_nodes_and_with_edges(self, cg):
        cg2 = cg.with_nodes(["Z"])
        assert "Z" in cg2.nodes()
        cg3 = cg.with_edges([("X", "Z")])
        assert ("X", "Z") in cg3.edges()

    def test_without_nodes_and_without_edges(self, cg):
        cg2 = cg.without_nodes(["U"])
        assert "U" not in cg2.nodes()
        cg3 = cg.without_edges([("U", "X")])
        assert ("U", "X") not in cg3.edges()
