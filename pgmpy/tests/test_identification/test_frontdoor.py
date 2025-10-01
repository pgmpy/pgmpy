import pytest

from pgmpy.base import DAG
from pgmpy.identification import Frontdoor


@pytest.fixture
def frontdoor_model():
    return DAG([("X", "M"), ("M", "Y")], roles={"exposure": "X", "outcome": "Y"})


@pytest.fixture
def frontdoor_model_latent():
    return DAG(
        [("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")],
        roles={"exposure": "X", "outcome": "Y"},
        latents={"U"},
    )


@pytest.fixture
def frontdoor_model_noniden():
    return DAG(
        [("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y"), ("U", "M")],
        roles={"exposure": "X", "outcome": "Y"},
        latents={"U"},
    )


def test_frontdoor(frontdoor_model):
    identified_dag, is_identified = Frontdoor().identify(frontdoor_model)

    assert is_identified is True
    assert identified_dag.get_role("exposure") == ["X"]
    assert identified_dag.get_role("outcome") == ["Y"]
    assert identified_dag.get_role("frontdoor") == ["M"]


def test_frontdoor_latent(frontdoor_model_latent):
    identified_dag, is_identified = Frontdoor().identify(frontdoor_model_latent)

    assert is_identified is True
    assert identified_dag.get_role("exposure") == ["X"]
    assert identified_dag.get_role("outcome") == ["Y"]
    assert identified_dag.get_role("frontdoor") == ["M"]
    assert identified_dag.latents == {"U"}


def test_frontdoor_latent_noniden(frontdoor_model_noniden):
    identified_dag, is_identified = Frontdoor().identify(frontdoor_model_noniden)

    assert is_identified is False
    assert identified_dag.get_role("exposure") == ["X"]
    assert identified_dag.get_role("outcome") == ["Y"]
    assert identified_dag.latents == {"U"}
