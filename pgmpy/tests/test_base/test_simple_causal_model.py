import networkx as nx

from pgmpy.base.simple_causal_model import SimpleCausalModel


def test_simple_string_variables():
    model = SimpleCausalModel(
        exposures="X", outcomes="Y", covariates="Z", mediators="M", instruments="I"
    )
    assert set(model.nodes()) == {"X", "Y", "Z", "M", "I"}
    assert set(model.edges()) == {
        ("X", "Y"),
        ("Z", "X"),
        ("Z", "Y"),
        ("I", "X"),
        ("X", "M"),
        ("M", "Y"),
    }
    assert set(model.get_role("exposure")) == {"X"}
    assert set(model.get_role("outcome")) == {"Y"}
    assert set(model.get_role("covariate")) == {"Z"}
    assert set(model.get_role("mediator")) == {"M"}
    assert set(model.get_role("instrument")) == {"I"}


def test_list_variables():
    model = SimpleCausalModel(
        exposures=["X1", "X2"],
        outcomes=["Y1", "Y2"],
        covariates=["Z"],
        mediators=["M"],
        instruments=["I"],
    )
    expected_edges = {
        ("X1", "Y1"),
        ("X1", "Y2"),
        ("X2", "Y1"),
        ("X2", "Y2"),
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
    assert set(model.get_role("exposure")) == {"X1", "X2"}
    assert set(model.get_role("outcome")) == {"Y1", "Y2"}
    assert set(model.get_role("covariate")) == {"Z"}
    assert set(model.get_role("mediator")) == {"M"}
    assert set(model.get_role("instrument")) == {"I"}


def test_integer_variables():
    model = SimpleCausalModel(
        exposures=1, outcomes=2, covariates=3, mediators=4, instruments=5
    )
    assert set(model.nodes()) == {"Var_1", "Var_2", "Var_3", "Var_4", "Var_5"}
    assert set(model.edges()) == {
        ("Var_1", "Var_2"),
        ("Var_3", "Var_1"),
        ("Var_3", "Var_2"),
        ("Var_5", "Var_1"),
        ("Var_1", "Var_4"),
        ("Var_4", "Var_2"),
    }
    assert set(model.get_role("exposure")) == {"Var_1"}
    assert set(model.get_role("outcome")) == {"Var_2"}
    assert set(model.get_role("covariate")) == {"Var_3"}
    assert set(model.get_role("mediator")) == {"Var_4"}
    assert set(model.get_role("instrument")) == {"Var_5"}


def test_missing_optional_args():
    model = SimpleCausalModel(exposures="X", outcomes="Y")
    assert set(model.edges()) == {("X", "Y")}
    assert set(model.get_role("exposure")) == {"X"}
    assert set(model.get_role("outcome")) == {"Y"}
    assert set(model.get_role("covariate")) == set()
    assert set(model.get_role("mediator")) == set()
    assert set(model.get_role("instrument")) == set()


def test_empty_covariates_mediators_instruments():
    model = SimpleCausalModel(
        exposures="X", outcomes="Y", covariates=None, mediators=None, instruments=[]
    )
    assert set(model.edges()) == {("X", "Y")}
    assert set(model.get_role("exposure")) == {"X"}
    assert set(model.get_role("outcome")) == {"Y"}
    assert set(model.get_role("covariate")) == set()
    assert set(model.get_role("mediator")) == set()
    assert set(model.get_role("instrument")) == set()


def test_multiple_exposures_outcomes():
    model = SimpleCausalModel(exposures=["X1", "X2"], outcomes=["Y1", "Y2"])
    expected_edges = {("X1", "Y1"), ("X1", "Y2"), ("X2", "Y1"), ("X2", "Y2")}
    assert set(model.edges()) == expected_edges
    assert set(model.get_role("exposure")) == {"X1", "X2"}
    assert set(model.get_role("outcome")) == {"Y1", "Y2"}


def test_latents():
    model = SimpleCausalModel(exposures="X", outcomes="Y", latents=["L"])
    assert "L" in model.latents


def test_is_dag():
    model = SimpleCausalModel(
        exposures="X", outcomes="Y", covariates="Z", mediators="M", instruments="I"
    )
    assert nx.is_directed_acyclic_graph(model)
