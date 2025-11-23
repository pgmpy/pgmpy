from pgmpy.base import DAG
from pgmpy.identification import InstrumentVariables


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
            "observed": ("x1", "y1", "x2", "y2"),
            "latents": ("xi1", "eta1"),
        },
    )
    iv = InstrumentVariables(variant=None)

    scaling_indicators = iv._get_scaling_indicators(model)

    assert scaling_indicators == {
        "xi1": "x1",
        "eta1": "y1",
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
            "observed": ("x1", "y1", "x2", "y2"),
            "latents": ("xi1", "eta1"),
        },
    )
    iv = InstrumentVariables(variant=None)

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


def test_get_ivs():
    iv_model = DAG(
        [("X", "Y"), ("I", "X"), ("U", "X"), ("U", "Y")],
        roles={
            "exposures": "X",
            "outcomes": "Y",
            "latents": "U",
        },
    )
    iv = InstrumentVariables(variant=None)
    graph_with_iv, ok = iv._identify(iv_model)
    assert ok is True
    expected_iv = {"I"}
    assert set(graph_with_iv.get_role("instrument")) == expected_iv
