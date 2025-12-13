import pytest

from pgmpy.base import DAG, SimpleCausalModel
from pgmpy.identification import InstrumentalVariables


def test_get_scaling_indicators():
    model = DAG(
        [
            ("xi1", "eta1"),
            ("xi1", "x1"),
            ("xi1", "x2"),
            ("eta1", "y1"),
            ("eta1", "y2"),
        ],
        roles={
            "exposures": "x1",
            "outcomes": "y1",
            "latents": ("xi1", "eta1"),
        },
    )
    iv = InstrumentalVariables(variant=None)

    scaling_indicators = iv._get_scaling_indicators(model)

    assert scaling_indicators == {
        "eta1": "y1",
        "xi1": "x1",
    }


def test_iv_transformations():
    model = DAG(
        [
            ("xi1", "eta1"),
            ("xi1", "x1"),
            ("xi1", "x2"),
            ("eta1", "y1"),
            ("eta1", "y2"),
        ],
        roles={
            "exposures": "x1",
            "outcomes": "y1",
            "latents": ("xi1", "eta1"),
        },
    )
    iv = InstrumentalVariables(variant=None)

    scaling_indicators = iv._get_scaling_indicators(model)

    transformed_graph, dependent_var = iv._iv_transformations(
        "xi1", "eta1", model, scaling_indicators=scaling_indicators
    )

    assert set(transformed_graph.edges()) == {
        ("xi1", "x1"),
        ("eta1", "y1"),
        ("xi1", "x2"),
        ("eta1", "y2"),
        ("eta1", "y1"),
        ("x1", "y1"),
    }

    assert dependent_var == "y1"


def test_get_ivs_without_scaling_indicators():
    iv_model = DAG(
        [("X", "Y"), ("I", "X"), ("U", "X"), ("U", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "latents": "U",
        },
    )
    iv = InstrumentalVariables(
        variant=None,
    )
    graph_with_iv, ok = iv._identify(iv_model)
    assert ok is True
    expected_iv = {"I"}
    assert set(graph_with_iv.get_role("instrument")) == expected_iv


def test_get_ivs_with_scaling_indicators():
    iv_model = DAG(
        [("X", "Y"), ("I", "X"), ("U", "X"), ("U", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "latents": "U",
        },
    )
    iv = InstrumentalVariables(variant=None, scaling_indicators={"U": "X"})
    graph_with_iv, ok = iv._identify(iv_model)
    assert ok is True
    expected_iv = {"I"}
    assert set(graph_with_iv.get_role("instrument")) == expected_iv


def test_get_ivs_with_multiple_latents():
    iv_model = DAG(
        [("X", "Y"), ("I", "X"), ("U", "X"), ("U", "Y"), ("U2", "X"), ("U2", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "latents": ("U", "U2"),
        },
    )
    iv = InstrumentalVariables(variant=None, scaling_indicators={"U": "X", "U2": "X"})
    graph_with_iv, ok = iv._identify(iv_model)
    assert ok is True
    expected_iv = {"I"}
    assert set(graph_with_iv.get_role("instrument")) == expected_iv


def test_get_ivs_when_scaling_indicator_incomplete():
    model = DAG(
        [
            ("xi1", "eta1"),
            ("xi1", "x1"),
            ("xi1", "x2"),
            ("eta1", "y1"),
            ("eta1", "y2"),
            ("eta2", "x3"),
            ("eta2", "y3"),
        ],
        roles={
            "exposures": "x1",
            "outcomes": "y1",
            "latents": ("xi1", "eta1", "eta2"),
        },
    )
    iv = InstrumentalVariables(variant=None, scaling_indicators={"xi1": "x1"})
    _ = iv._get_scaling_indicators(model)
    # assert the returned scaling indicators are correct
    assert iv.scaling_indicators == {"xi1": "x1", "eta1": "y1", "eta2": "x3"}


def test_no_ivs_found():
    iv_model = DAG(
        [("X", "Y"), ("U", "X"), ("U", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "latents": "U",
            "observed": ("X", "Y"),
        },
    )
    iv = InstrumentalVariables(variant=None)
    retruned_graph, ok = iv._identify(iv_model)
    assert ok is False
    expected_iv = set()
    assert set(retruned_graph.get_role("instrument")) == expected_iv
    assert not retruned_graph.has_role("instrument")


def test_conditional_ivs():
    conditional_iv_model = DAG(
        [("I", "X"), ("X", "Y"), ("W", "I"), ("W", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "observed": ("X", "Y", "I", "W"),
        },
    )
    iv = iv = InstrumentalVariables(variant="conditional")
    retruned_graph, ok = iv._identify(conditional_iv_model)
    assert ok is True
    expected_iv = "I"
    expected_conditional = "W"
    assert (retruned_graph.get_role("instrument")) == list(expected_iv)
    assert (retruned_graph.get_role("conditional")) == list(expected_conditional)


def test_conditional_ivs_with_latents():
    conditional_iv_model = DAG(
        [("U", "X"), ("U", "Y"), ("I", "X"), ("X", "Y"), ("W", "I"), ("W", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "latents": "U",
            "observed": ("X", "Y", "I", "W"),
        },
    )
    iv = iv = InstrumentalVariables(variant="conditional")
    returned_graph, ok = iv._identify(conditional_iv_model)
    assert ok is True
    expected_iv = "I"
    expected_conditonal = "W"
    assert (returned_graph.get_role("instrument")) == list(expected_iv)
    assert (returned_graph.get_role("conditional")) == list(expected_conditonal)


def test_no_ivs_with_SCM():
    model = SimpleCausalModel(exposures="X", covariates="U", outcomes="Y", latents="U")
    iv = InstrumentalVariables(variant=None)
    retruned_graph, ok = iv._identify(model)
    assert ok is False
    expected_iv = set()
    assert set(retruned_graph.get_role("instrument")) == expected_iv
    assert not retruned_graph.has_role("instrument")


def test_usage_with_SCM():
    model = SimpleCausalModel(exposures="X", covariates="U", outcomes="Y", latents="U")
    model.add_edge("I", "X")
    iv = InstrumentalVariables(variant=None)
    retruned_graph, ok = iv._identify(model)
    assert ok is True
    expected_iv = {"I"}
    assert set(retruned_graph.get_role("instrument")) == expected_iv
    assert retruned_graph.has_role("instrument")


def test_usage_with_SCM_with_mediator():
    model = SimpleCausalModel(
        exposures="X", covariates="U", outcomes="Y", mediators="M", latents="U"
    )
    model.add_edge("I", "X")
    iv = InstrumentalVariables(variant=None)
    assert pytest.raises(ValueError, iv._identify, model)


def test_validate_causal_graph():
    iv_model = DAG(
        [("X", "Y"), ("I", "X"), ("U", "X"), ("U", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "latents": "U",
            "instrument": "I",
        },
    )
    iv = InstrumentalVariables(variant=None)
    ok = iv._validate(iv_model)
    assert ok is True


def test_validate_causal_graph_returns_false():
    incorrect_iv_model = DAG(
        [("X", "Y"), ("I", "X"), ("U", "X"), ("U", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "latents": "U",
            "instrument": "X",
        },
    )
    iv = InstrumentalVariables(variant=None)
    ok = iv._validate(incorrect_iv_model)
    assert ok is False


def test_validate_causal_graph_with_conditional_ivs():
    conditional_iv_model = DAG(
        [("I", "X"), ("X", "Y"), ("W", "I"), ("W", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "observed": ("X", "Y", "I", "W"),
            "instrument": ("I"),
            "conditional": ("W"),
        },
    )
    iv = InstrumentalVariables(variant="conditional")
    ok = iv._validate(conditional_iv_model)
    assert ok is True


def test_validate_when_no_ivs():
    no_iv_model = DAG(
        [("I", "X"), ("X", "Y"), ("W", "I"), ("W", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "observed": ("X", "Y", "I", "W"),
            "conditional": ("W"),
        },
    )
    iv = InstrumentalVariables(variant=None)
    assert pytest.raises(ValueError, iv._validate, no_iv_model)


def test_validate_when_no_conditionals():
    no_conditional_model = DAG(
        [("I", "X"), ("X", "Y"), ("W", "I"), ("W", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "observed": ("X", "Y", "I", "W"),
            "instrument": ("I"),
        },
    )
    iv = InstrumentalVariables(variant=None)
    assert pytest.raises(ValueError, iv._validate, no_conditional_model)
