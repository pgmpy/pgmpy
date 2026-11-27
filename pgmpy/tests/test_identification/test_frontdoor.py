import pytest

from pgmpy.base import DAG
from pgmpy.identification import Frontdoor


@pytest.fixture
def frontdoor_model():
    return DAG([("X", "M"), ("M", "Y")], roles={"exposures": "X", "outcomes": "Y"})


@pytest.fixture
def frontdoor_model_latent():
    return DAG(
        [("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")],
        roles={"exposures": "X", "outcomes": "Y"},
        latents={"U"},
    )


@pytest.fixture
def frontdoor_model_noniden():
    return DAG(
        [("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y"), ("U", "M")],
        roles={"exposures": "X", "outcomes": "Y"},
        latents={"U"},
    )


def test_frontdoor(frontdoor_model):
    identified_dag, is_identified = Frontdoor().identify(frontdoor_model)

    assert is_identified is True
    assert identified_dag.get_role("exposures") == ["X"]
    assert identified_dag.get_role("outcomes") == ["Y"]
    assert identified_dag.get_role("frontdoor") == ["M"]


def test_frontdoor_latent(frontdoor_model_latent):
    identified_dag, is_identified = Frontdoor().identify(frontdoor_model_latent)

    assert is_identified is True
    assert identified_dag.get_role("exposures") == ["X"]
    assert identified_dag.get_role("outcomes") == ["Y"]
    assert identified_dag.get_role("frontdoor") == ["M"]
    assert identified_dag.latents == {"U"}


def test_frontdoor_latent_noniden(frontdoor_model_noniden):
    identified_dag, is_identified = Frontdoor().identify(frontdoor_model_noniden)

    assert is_identified is False
    assert identified_dag.get_role("exposures") == ["X"]
    assert identified_dag.get_role("outcomes") == ["Y"]
    assert identified_dag.latents == {"U"}


def test_frontdoor_validate_all_exposure_outcome_pairs():
    """Validation must consider every exposure-outcome pair, not only the first (#3754)."""
    dag = DAG(
        [
            ("X1", "M"),
            ("M", "Y1"),
            ("U", "X1"),
            ("U", "Y1"),
            ("X2", "M"),
            ("M", "Y2"),
            ("X2", "Y2"),
        ],
        roles={"exposures": ["X1", "X2"], "outcomes": ["Y1", "Y2"]},
        latents={"U"},
    )
    dag_with_frontdoor = dag.with_role("frontdoor", ["M"], inplace=False)

    assert Frontdoor().validate(dag_with_frontdoor) is False
