import networkx as nx
import pytest

from pgmpy.base import SimpleCausalModel


def test_simple_string_variables():
    model = SimpleCausalModel(
        exposures="X", outcomes="Y", confounders="Z", mediators="M", instruments="I"
    )
    assert set(model.nodes()) == {"X", "Y", "Z", "M", "I"}
    expected_edges = {("Z", "X"), ("Z", "Y"), ("I", "X"), ("X", "M"), ("M", "Y")}
    assert set(model.edges()) == expected_edges
    assert set(model.get_role("exposures")) == {"X"}
    assert set(model.get_role("outcomes")) == {"Y"}
    assert set(model.get_role("confounders")) == {"Z"}
    assert set(model.get_role("mediators")) == {"M"}
    assert set(model.get_role("instruments")) == {"I"}


def test_list_variables():
    model = SimpleCausalModel(
        exposures=["X1", "X2"],
        outcomes=["Y1", "Y2"],
        confounders=["Z"],
        mediators=["M"],
        instruments=["I"],
    )
    expected_edges = {
        ("Z", "X1"),
        ("Z", "X2"),
        ("Z", "Y1"),
        ("Z", "Y2"),
        ("I", "X1"),
        ("I", "X2"),
        ("X1", "M"),
        ("X2", "M"),
        ("M", "Y1"),
        ("M", "Y2"),
    }
    assert set(model.edges()) == expected_edges
    assert set(model.get_role("exposures")) == {"X1", "X2"}
    assert set(model.get_role("outcomes")) == {"Y1", "Y2"}
    assert set(model.get_role("confounders")) == {"Z"}
    assert set(model.get_role("mediators")) == {"M"}
    assert set(model.get_role("instruments")) == {"I"}


def test_integer_variables():
    model = SimpleCausalModel(
        exposures=1,
        outcomes=2,
        confounders=3,
        mediators=4,
        instruments=5,
        latents=["X_0", "X_1"],
    )
    expected_nodes = {
        "E_0",
        "O_0",
        "O_1",
        "X_0",
        "X_1",
        "X_2",
        "M_0",
        "M_1",
        "M_2",
        "M_3",
        "I_0",
        "I_1",
        "I_2",
        "I_3",
        "I_4",
    }
    assert set(model.nodes()) == expected_nodes
    assert ("E_0", "O_0") not in set(model.edges())
    assert ("E_0", "O_1") not in set(model.edges())
    assert ("X_0", "E_0") in set(model.edges())
    assert ("X_0", "O_0") in set(model.edges())
    assert ("X_0", "O_1") in set(model.edges())
    assert ("I_0", "E_0") in set(model.edges())
    assert ("E_0", "M_0") in set(model.edges())
    assert ("M_0", "O_0") in set(model.edges())
    assert set(model.get_role("exposures")) == {"E_0"}
    assert set(model.get_role("outcomes")) == {"O_0", "O_1"}
    assert set(model.get_role("confounders")) == {"X_0", "X_1", "X_2"}
    assert set(model.get_role("mediators")) == {"M_0", "M_1", "M_2", "M_3"}
    assert set(model.get_role("instruments")) == {"I_0", "I_1", "I_2", "I_3", "I_4"}
    assert set(model.latents) == {"X_0", "X_1"}


def test_missing_optional_args():
    model = SimpleCausalModel(exposures="X", outcomes="Y")
    assert set(model.edges()) == {("X", "Y")}
    assert set(model.get_role("exposures")) == {"X"}
    assert set(model.get_role("outcomes")) == {"Y"}
    assert set(model.get_role("confounders")) == set()
    assert set(model.get_role("mediators")) == set()
    assert set(model.get_role("instruments")) == set()


def test_empty_confounders_mediators_instruments():
    model = SimpleCausalModel(
        exposures="X", outcomes="Y", confounders=None, mediators=None, instruments=[]
    )
    assert set(model.edges()) == {("X", "Y")}
    assert set(model.get_role("exposures")) == {"X"}
    assert set(model.get_role("outcomes")) == {"Y"}
    assert set(model.get_role("confounders")) == set()
    assert set(model.get_role("mediators")) == set()
    assert set(model.get_role("instruments")) == set()


def test_multiple_exposures_outcomes():
    model = SimpleCausalModel(exposures=["X1", "X2"], outcomes=["Y1", "Y2"])
    expected_edges = {("X1", "Y1"), ("X1", "Y2"), ("X2", "Y1"), ("X2", "Y2")}
    assert set(model.edges()) == expected_edges
    assert set(model.get_role("exposures")) == {"X1", "X2"}
    assert set(model.get_role("outcomes")) == {"Y1", "Y2"}


def test_latents():
    with pytest.raises(ValueError):
        SimpleCausalModel(exposures="X", outcomes="Y", latents=["L"])

    model = SimpleCausalModel(
        exposures="X", outcomes="Y", confounders="Z", latents=["Z"]
    )
    assert set(model.nodes()) == {"X", "Y", "Z"}
    assert set(model.latents) == {"Z"}


def test_is_dag():
    model = SimpleCausalModel(
        exposures="X", outcomes="Y", confounders="Z", mediators="M", instruments="I"
    )
    assert nx.is_directed_acyclic_graph(model)
