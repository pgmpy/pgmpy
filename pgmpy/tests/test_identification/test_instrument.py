from pgmpy.base import DAG
from pgmpy.identification import InstrumentalVariables


def test_get_scaling_indicators():

    model = DAG(
        [
            ("xi1", "eta1"),
            ("xi1", "eta1"),
            ("xi1", "x1"),
            ("xi1", "x2"),
            ("eta1", "y1"),
            ("eta1", "y2"),
        ],
        roles={
            "exposures": "x1",
            "outcomes": "y1",
            # "observed": ("x1", "y1", "x2", "y2"),
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
            # "observed": ("x1", "y1", "x2", "y2"),
            "latents": ("xi1", "eta1"),
        },
    )
    iv = InstrumentalVariables(variant=None)

    scaling_indicators = iv._get_scaling_indicators(model)

    transformed_graph, dependent_var = iv._iv_transformations(
        "xi1", "eta1", model, scaling_indicators=scaling_indicators
    )

    assert set(transformed_graph.edges()) == {
        #     (".eta1", "y1"),
        #     (".xi1", "x1"),
        #     (".xi1", "x2"),
        # }
        ("xi1", "x1"),
        (".eta1", "y1"),
        ("xi1", "x2"),
        ("eta1", "y2"),
        ("eta1", "y1"),
        (".x1", "y1"),
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


def test_no_ivs_found():
    iv_model = DAG(
        [("X", "Y"), ("U", "X"), ("U", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "latents": "U",
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
        },
    )
    iv = iv = InstrumentalVariables(variant="conditional")
    retruned_graph, ok = iv._identify(conditional_iv_model)
    assert ok is True
    expected_iv = ("I", "W")
    assert (retruned_graph.get_role("instrument")) == list(expected_iv)


def test_conditional_ivs_with_latents():
    conditional_iv_model = DAG(
        [("U", "X"), ("U", "Y"), ("I", "X"), ("X", "Y"), ("W", "I"), ("W", "Y")],
        roles={"exposures": "X", "outcomes": "Y", "latents": "U"},
    )
    iv = iv = InstrumentalVariables(variant="conditional")
    retruned_graph, ok = iv._identify(conditional_iv_model)
    assert ok is True
    expected_iv = ("I", "W")
    assert (retruned_graph.get_role("instrument")) == list(expected_iv)
