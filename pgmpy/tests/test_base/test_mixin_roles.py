# pgmpy/tests/test_base/test_mixin_roles.py
import pytest

from pgmpy.base import DAG


@pytest.fixture
def basic_dag():
    G = DAG(ebunch=[("X", "Y"), ("Z", "Y")])
    G.add_node("U")
    return G


def test_with_role_single_variable(basic_dag):
    basic_dag.with_role(role="exposures", variables="X", inplace=True)

    assert "roles" in basic_dag.nodes["X"]
    assert basic_dag.nodes["X"]["roles"] == {"exposures"}
    assert set(basic_dag.get_role("exposures")) == {"X"}
    assert basic_dag.has_role("exposures") is True


def test_with_role_multiple_variables(basic_dag):
    basic_dag.with_role(role="outcomes", variables={"Y", "Z"}, inplace=True)

    assert basic_dag.nodes["Y"]["roles"] == {"outcomes"}
    assert basic_dag.nodes["Z"]["roles"] == {"outcomes"}
    assert set(basic_dag.get_role("outcomes")) == {"Y", "Z"}


def test_with_role_adds_without_overwriting_existing_roles(basic_dag):
    basic_dag.with_role(role="exposures", variables="X", inplace=True)
    basic_dag.with_role(role="outcomes", variables={"X", "Y"}, inplace=True)

    assert basic_dag.nodes["X"]["roles"] == {"exposures", "outcomes"}
    assert basic_dag.nodes["Y"]["roles"] == {"outcomes"}


def test_with_role_raises_for_missing_variable(basic_dag):
    with pytest.raises(ValueError):
        basic_dag.with_role(role="exposures", variables="MISSING", inplace=True)


def test_with_role_inplace_false_returns_new_graph(basic_dag):
    new_graph = basic_dag.with_role(role="exposures", variables={"X", "Z"}, inplace=False)

    for _, attr in basic_dag.nodes(data=True):
        assert "roles" not in attr

    assert new_graph is not basic_dag

    assert new_graph.nodes["X"]["roles"] == {"exposures"}
    assert new_graph.nodes["Z"]["roles"] == {"exposures"}
    assert set(new_graph.get_role("exposures")) == {"X", "Z"}


def test_inplace_argument(basic_dag):
    # inplace=True returns None and mutates the current graph.
    result = basic_dag.with_role(role="exposures", variables="X", inplace=True)
    assert result is None
    assert "exposures" in basic_dag.nodes["X"]["roles"]

    result = basic_dag.without_role(role="exposures", variables="X", inplace=True)
    assert result is None
    assert "roles" not in basic_dag.nodes["X"]

    # inplace=False returns a new graph and does not mutate the current graph.
    new_graph = basic_dag.with_role(role="outcomes", variables="X", inplace=False)
    assert new_graph is not None
    assert new_graph is not basic_dag
    assert "outcomes" in new_graph.nodes["X"]["roles"]
    assert "roles" not in basic_dag.nodes["X"]

    basic_dag.with_role(role="exposures", variables="X", inplace=True)
    new_graph = basic_dag.without_role(role="exposures", variables="X", inplace=False)
    assert new_graph is not None
    assert new_graph is not basic_dag
    assert "roles" not in new_graph.nodes["X"]
    assert "exposures" in basic_dag.nodes["X"]["roles"]


def test_get_roles_and_get_role_dict(basic_dag):
    basic_dag.with_role(role="exposures", variables="X", inplace=True)
    basic_dag.with_role(role="outcomes", variables={"Y", "Z"}, inplace=True)
    basic_dag.with_role(role="latents", variables="U", inplace=True)

    roles = set(basic_dag.get_roles())
    assert roles == {"exposures", "outcomes", "latents"}

    role_dict = basic_dag.get_role_dict()
    assert set(role_dict.keys()) == roles
    assert set(role_dict["exposures"]) == {"X"}
    assert set(role_dict["outcomes"]) == {"Y", "Z"}
    assert set(role_dict["latents"]) == {"U"}


def test_without_role_specific_variables(basic_dag):
    basic_dag.with_role("exposures", {"X", "Z"}, inplace=True)
    basic_dag.with_role("outcomes", {"Y"}, inplace=True)

    basic_dag.without_role(role="exposures", variables="X", inplace=True)

    assert "exposures" not in basic_dag.nodes["X"].get("roles", set())
    assert "exposures" in basic_dag.nodes["Z"]["roles"]
    assert "outcomes" in basic_dag.nodes["Y"]["roles"]


def test_without_role_removes_roles_attr_when_last_role_removed(basic_dag):
    basic_dag.with_role("exposures", "X", inplace=True)

    # After removing the only role, "roles" attribute should disappear
    basic_dag.without_role(role="exposures", variables="X", inplace=True)
    assert "roles" not in basic_dag.nodes["X"]


def test_without_role_all_variables_when_variables_none(basic_dag):
    basic_dag.with_role("exposures", {"X", "Z"}, inplace=True)
    basic_dag.with_role("outcomes", {"Y"}, inplace=True)

    basic_dag.without_role(role="exposures", variables=None, inplace=True)
    assert "exposures" not in basic_dag.nodes["X"].get("roles", set())
    assert "exposures" not in basic_dag.nodes["Z"].get("roles", set())
    assert "outcomes" in basic_dag.nodes["Y"]["roles"]


def test_without_role_inplace_false_returns_new_graph(basic_dag):
    basic_dag.with_role("exposures", {"X", "Z"}, inplace=True)
    basic_dag.with_role("outcomes", {"Y"}, inplace=True)

    new_graph = basic_dag.without_role(role="exposures", variables="X", inplace=False)

    assert "exposures" in basic_dag.nodes["X"]["roles"]
    assert "exposures" in basic_dag.nodes["Z"]["roles"]

    assert new_graph is not basic_dag

    assert "exposures" not in new_graph.nodes["X"].get("roles", set())
    assert "exposures" in new_graph.nodes["Z"]["roles"]
    assert "outcomes" in new_graph.nodes["Y"]["roles"]


def test_latents_property_and_observed():
    G = DAG(ebunch=[("a", "b")], latents="a")

    assert G.latents == {"a"}
    assert G.observed == {"b"}

    # Setting latents again should replace the old latent set
    G.latents = {"b"}
    assert G.latents == {"b"}
    assert G.observed == {"a"}


def test_observed_when_no_latents():
    G = DAG(ebunch=[("a", "b")])
    assert G.latents == set()
    assert G.observed == {"a", "b"}


def test_exposures_and_outcomes_properties():
    G = DAG(ebunch=[("X", "Y")])

    G.exposures = "X"
    G.outcomes = {"Y"}

    assert G.exposures == {"X"}
    assert G.outcomes == {"Y"}

    # Changing exposures should replace the previous exposures role
    G.exposures = {"Y"}
    assert G.exposures == {"Y"}
    assert G.outcomes == {"Y"}


def test_is_valid_causal_structure_raises_when_missing_roles():
    G = DAG(ebunch=[("X", "Y")])

    # No roles at all
    with pytest.raises(ValueError):
        G.is_valid_causal_structure()

    # Only exposures
    G.exposures = "X"
    with pytest.raises(ValueError):
        G.is_valid_causal_structure()

    # Only outcomes
    G = DAG(ebunch=[("X", "Y")])
    G.outcomes = "Y"
    with pytest.raises(ValueError):
        G.is_valid_causal_structure()


def test_is_valid_causal_structure_passes_with_exposures_and_outcomes():
    G = DAG(ebunch=[("X", "Y")])
    G.exposures = "X"
    G.outcomes = "Y"

    assert G.is_valid_causal_structure() is True
