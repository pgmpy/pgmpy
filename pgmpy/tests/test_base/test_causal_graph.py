import pytest

from pgmpy.base import DAG as DAG


@pytest.fixture
def cg():
    edges = [("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")]
    roles = {"exposure": "X", "outcome": "Y"}
    return DAG(ebunch=edges, roles=roles)


@pytest.fixture
def cg2():
    cg2 = DAG(
        ebunch=[("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")],
        roles={"adjustment": {"U", "M"}, "exposure": "X"},
    )
    return cg2


@pytest.fixture
def edges():
    return [("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")]


class TestDAG:

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
        assert not cg3.has_role("adjustment")

    def test_get_roles(self, edges):
        cg = DAG(
            ebunch=edges,
            roles={"exposure": "X", "outcome": "Y", "adjustment": {"U", "M"}},
        )
        roles = cg.get_roles()
        assert set(roles) == {"exposure", "outcome", "adjustment"}

    def test_get_role_dict(self, cg):
        role_dict = cg.get_role_dict()
        assert role_dict == {"exposure": ["X"], "outcome": ["Y"]}

    def test_get_role_dict2(self, cg2):
        role_dict = cg2.get_role_dict()
        assert role_dict == {"adjustment": ["U", "M"], "exposure": ["X"]}

    def test_with_role_invalid_variable(self, cg):
        with pytest.raises(ValueError):
            cg.with_role("adjustment", {"Z"})

    @pytest.mark.xfail(reason="Equality not implemented for DAG")
    def test_copy_and_equality(self, cg):
        cg2 = cg.copy()
        assert cg == cg2
        cg3 = cg.with_role("adjustment", {"U"})
        assert cg != cg3

    @pytest.mark.xfail(reason="Hashing not implemented for DAG")
    def test_hash(self, cg):
        cg2 = cg.copy()
        assert hash(cg) == hash(cg2)
        cg3 = cg.with_role("adjustment", {"U"})
        assert hash(cg) != hash(cg3)

    def test_is_valid_causal_structure(self, cg):
        assert cg.is_valid_causal_structure()
        cg2 = DAG(
            ebunch=[("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")],
            roles={"target": "Y", "exposure": {"M", "X"}},
        )
        with pytest.raises(ValueError):
            cg2.is_valid_causal_structure()
