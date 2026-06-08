"""
Tests for the sklearn-compatible LLMPairwise class in pgmpy.causal_discovery
"""

import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

from pgmpy.causal_discovery import LLMPairwise


def install_fake_litellm(monkeypatch, content):
    """Install a fake ``litellm`` module that returns ``content`` from ``completion``.

    The fake module records the keyword arguments it was called with on
    ``fake_litellm.kwargs`` so tests can assert on what was passed through.
    """
    fake_litellm = types.ModuleType("litellm")

    def completion(**kwargs):
        fake_litellm.kwargs = kwargs
        return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])

    fake_litellm.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    return fake_litellm


@pytest.fixture
def pair_data():
    """A simple two-variable dataset used across the orientation tests."""
    return pd.DataFrame(
        {
            "Smoker": [0, 1, 1, 0],
            "Cancer": [0, 1, 0, 0],
        }
    )


def test_fit_orients_first_column_to_second(monkeypatch, pair_data):
    """A response of "1" orients the edge from the first to the second column."""
    fake_litellm = install_fake_litellm(monkeypatch, "1")
    est = LLMPairwise(
        descriptions={
            "Smoker": "Whether a person smokes",
            "Cancer": "Whether a person has cancer",
        },
        llm_kwargs={"temperature": 0},
    ).fit(pair_data)

    assert ("Smoker", "Cancer") in est.causal_graph_.edges()
    assert est.direction_score_ == 1.0
    assert est.response_ == "1"
    assert "Whether a person smokes" in est.prompt_
    assert "Whether a person has cancer" in est.prompt_
    assert fake_litellm.kwargs["model"] == "gemini/gemini-1.5-flash"
    assert fake_litellm.kwargs["messages"] == [{"role": "user", "content": est.prompt_}]
    assert fake_litellm.kwargs["temperature"] == 0


def test_fit_orients_second_column_to_first(monkeypatch, pair_data):
    """A response of "2" orients the edge from the second to the first column."""
    install_fake_litellm(monkeypatch, "2")
    est = LLMPairwise().fit(pair_data)

    assert ("Cancer", "Smoker") in est.causal_graph_.edges()
    assert est.direction_score_ == -1.0
    assert est.adjacency_matrix_.loc["Cancer", "Smoker"] == 1
    assert est.adjacency_matrix_.loc["Smoker", "Cancer"] == 0


def test_fit_uses_variable_name_as_missing_description(monkeypatch, pair_data):
    """Variables without a description fall back to their column name in the prompt."""
    install_fake_litellm(monkeypatch, "1")
    est = LLMPairwise(descriptions={"Smoker": "Whether a person smokes"}).fit(pair_data)

    assert "<A>: Whether a person smokes" in est.prompt_
    assert "<B>: Cancer" in est.prompt_


def test_fit_accepts_categorical_data(monkeypatch):
    """Categorical input is not blocked since orientation uses names, not values."""
    install_fake_litellm(monkeypatch, "1")
    data = pd.DataFrame(
        {
            "Treatment": pd.Categorical(["yes", "no", "yes"]),
            "Outcome": pd.Categorical(["high", "low", "high"]),
        }
    )

    est = LLMPairwise().fit(data)

    assert ("Treatment", "Outcome") in est.causal_graph_.edges()


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"A": [0, 1, 2]}),
        pd.DataFrame({"A": [0, 1, 2], "B": [1, 2, 3], "C": [2, 3, 4]}),
    ],
)
def test_fit_requires_exactly_two_columns(monkeypatch, data):
    """Fitting on anything other than two variables raises a ValueError."""
    install_fake_litellm(monkeypatch, "1")
    est = LLMPairwise()

    with pytest.raises(ValueError, match="requires exactly two variables"):
        est.fit(data)


def test_fit_raises_for_unclear_response(monkeypatch, pair_data):
    """A response that is neither option raises a ValueError."""
    install_fake_litellm(monkeypatch, "unclear")
    est = LLMPairwise()

    with pytest.raises(ValueError, match="Results from the LLM are unclear"):
        est.fit(pair_data)
