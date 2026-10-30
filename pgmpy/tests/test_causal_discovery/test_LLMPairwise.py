import importlib
import os
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import LLMPairwise


@pytest.fixture
def data():
    return pd.DataFrame({"Smoker": [0, 1, 1, 0], "Cancer": [0, 1, 0, 0]})


@pytest.mark.skipif("GEMINI_API_KEY" not in os.environ, reason="Gemini API key is not set")
def test_llm_api():
    """Integration test that actually queries the LLM (requires GEMINI_API_KEY)."""
    descriptions = {
        "Age": "The age of a person",
        "Income": "The income i.e. amount of money the person makes",
    }
    df = pd.DataFrame({"Age": [0, 1, 1, 0], "Income": [0, 1, 0, 1]})

    assert ("Age", "Income") in LLMPairwise(descriptions=descriptions).fit(df).causal_graph_.edges()
    assert ("Age", "Income") in LLMPairwise(descriptions=descriptions).fit(df[["Income", "Age"]]).causal_graph_.edges()


def expected_failed_checks(estimator):
    # LLMPairwise orients a single pair, so checks that fit on a different
    # number of columns cannot apply.
    checks = dict.fromkeys(
        (
            "check_fit_score_takes_y",
            "check_dont_overwrite_parameters",
            "check_n_features_in_after_fitting",
            "check_positive_only_tag_during_fit",
            "check_estimators_dtypes",
            "check_dtype_object",
            "check_pipeline_consistency",
            "check_estimators_nan_inf",
            "check_estimators_pickle",
            "check_f_contiguous_array_estimator",
            "check_methods_sample_order_invariance",
            "check_methods_subset_invariance",
            "check_fit2d_1feature",
            "check_dict_unchanged",
            "check_fit2d_predict1d",
        ),
        "LLMPairwise orients exactly two variables; this check fits on a different number of columns.",
    )
    # `fit` resolves the default system_prompt onto the instance, which this
    # check flags as mutating an __init__ parameter.
    checks["check_estimators_overwrite_params"] = (
        "LLMPairwise sets the default system_prompt on the instance during fit."
    )
    return checks


@parametrize_with_checks([LLMPairwise()], expected_failed_checks=expected_failed_checks)
def test_llmpairwise_compatibility(estimator, check, monkeypatch):
    monkeypatch.setattr(LLMPairwise, "_query_llm", lambda self, messages: "1")
    check(estimator)


def test_fit(monkeypatch, data):
    monkeypatch.setattr(LLMPairwise, "_query_llm", lambda self, messages: "1")
    est = LLMPairwise().fit(data)
    assert ("Smoker", "Cancer") in est.causal_graph_.edges()
    assert est.adjacency_matrix_.loc["Smoker", "Cancer"] == 1

    monkeypatch.setattr(LLMPairwise, "_query_llm", lambda self, messages: "2")
    est = LLMPairwise().fit(data)
    assert ("Cancer", "Smoker") in est.causal_graph_.edges()

    for frame in (
        pd.DataFrame({"A": [0, 1, 2]}),
        pd.DataFrame({"A": [0, 1], "B": [1, 2], "C": [2, 3]}),
    ):
        with pytest.raises(ValueError, match="requires exactly two variables"):
            LLMPairwise().fit(frame)


def test_use_cache(monkeypatch, data):
    calls = {"n": 0}

    def fake_query(self, messages):
        calls["n"] += 1
        return "1"

    monkeypatch.setattr(LLMPairwise, "_query_llm", fake_query)

    # With caching, re-fitting the same pair reuses the result (no second query).
    est = LLMPairwise(use_cache=True)
    est.fit(data)
    est.fit(data)
    assert calls["n"] == 1
    assert ("Smoker", "Cancer") in est.causal_graph_.edges()

    # Without caching, every fit queries the LLM again.
    calls["n"] = 0
    est = LLMPairwise(use_cache=False)
    est.fit(data)
    est.fit(data)
    assert calls["n"] == 2


def test_build_prompt():
    est = LLMPairwise(
        descriptions={"Smoker": "Whether a person smokes"},
        system_prompt="You are a careful causal reasoner.",
    )
    system, user = est._build_prompt("Smoker", "Cancer")

    assert (system["role"], user["role"]) == ("system", "user")
    assert system["content"] == "You are a careful causal reasoner."
    assert "Whether a person smokes" in user["content"]
    assert "<B>: Cancer" in user["content"]


def test_query_llm(monkeypatch):
    module = importlib.import_module(LLMPairwise.__module__)
    calls = {}

    def completion(**kwargs):
        calls.update(kwargs)
        return MagicMock(choices=[MagicMock(message=MagicMock(content="1"))])

    monkeypatch.setattr(module, "litellm", types.SimpleNamespace(completion=completion))
    messages = [{"role": "user", "content": "hi"}]
    response = LLMPairwise(llm_kwargs={"temperature": 0})._query_llm(messages)

    assert response == "1"
    assert calls == {"model": "gemini/gemini-2.5-flash", "messages": messages, "temperature": 0}


def test_parse_response():
    est = LLMPairwise()
    for response, expected in (
        ("1.", ("Smoker", "Cancer")),
        ("Option 1", ("Smoker", "Cancer")),
        ("**1**", ("Smoker", "Cancer")),
        ("A", ("Smoker", "Cancer")),
        ("Answer: 2", ("Cancer", "Smoker")),
        ("B", ("Cancer", "Smoker")),
    ):
        assert est._parse_response(response, "Smoker", "Cancer") == expected

    with pytest.raises(ValueError, match="unclear"):
        est._parse_response("no idea", "Smoker", "Cancer")
