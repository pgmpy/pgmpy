import os
import random
import sys
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from tqdm.auto import tqdm

from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.utils import (
    discretize,
    get_example_model,
    llm_pairwise_orient,
    preprocess_data,
)
from pgmpy.utils.utils import (
    _build_pairwise_orient_prompt,
    _parse_pairwise_orient_response,
    _transformers_pipeline_cache,
)


class TestDiscretization(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal(1000)
        Y = 0.2 * X + rng.standard_normal(1000)
        Z = 0.4 * X + 0.5 * Y + rng.standard_normal(1000)

        self.data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_rounding_disc(self):
        df_disc = discretize(data=self.data, cardinality={"X": 5, "Y": 4, "Z": 3}, method="rounding")
        self.assertEqual(df_disc["X"].nunique(), 5)
        self.assertEqual(df_disc["Y"].nunique(), 4)
        self.assertEqual(df_disc["Z"].nunique(), 3)

        df_disc = discretize(data=self.data, cardinality={"X": 5, "Y": 4, "Z": 3}, method="quantile")
        self.assertEqual(df_disc["X"].nunique(), 5)
        self.assertEqual(df_disc["Y"].nunique(), 4)
        self.assertEqual(df_disc["Z"].nunique(), 3)


class TestPairwiseOrientation(unittest.TestCase):
    """Tests for `llm_pairwise_orient` and its pluggable backends.

    The test matrix exercises: prompt construction, response parsing (strict
    and tolerant), callable backend, each built-in backend via mocking, and
    backward-compatibility with the pre-backend call signature.
    """

    @pytest.mark.skipif("GEMINI_API_KEY" not in os.environ, reason="Gemini API key is not set")
    def test_llm(self):
        descriptions = {
            "Age": "The age of a person",
            "Workclass": "The workplace where the person is employed such as Private industry, or self employed",
            "Education": "The highest level of education the person has finished",
            "MaritalStatus": "The marital status of the person",
            "Occupation": "The kind of job the person does. For example, sales, craft repair, clerical",
            "Relationship": "The relationship status of the person",
            "Race": "The ethnicity of the person",
            "Sex": "The sex or gender of the person",
            "HoursPerWeek": "The number of hours per week the person works",
            "NativeCountry": "The native country of the person",
            "Income": "The income i.e. amount of money the person makes",
        }

        self.assertEqual(
            llm_pairwise_orient(x="Age", y="Income", descriptions=descriptions, domain="Social Sciences"),
            ("Age", "Income"),
        )
        self.assertEqual(
            llm_pairwise_orient(x="Income", y="Age", descriptions=descriptions, domain="Social Sciences"),
            ("Age", "Income"),
        )

    # --- Prompt construction ---

    def setUp(self):
        self.descriptions = {"Age": "age of a person", "Income": "income of a person"}

    def test_build_prompt_contains_descriptions(self):
        prompt = _build_pairwise_orient_prompt("Age", "Income", self.descriptions, None)
        self.assertIn("age of a person", prompt)
        self.assertIn("income of a person", prompt)
        self.assertIn("<A>", prompt)
        self.assertIn("<B>", prompt)
        # Default system prompt should appear
        self.assertIn("expert in Causal Inference", prompt)

    def test_build_prompt_custom_system_prompt(self):
        prompt = _build_pairwise_orient_prompt(
            "Age", "Income", self.descriptions, system_prompt="You are a clinician"
        )
        self.assertIn("You are a clinician", prompt)

    # --- Response parsing ---

    def test_parse_strict_1(self):
        self.assertEqual(_parse_pairwise_orient_response("1", "X", "Y"), ("X", "Y"))

    def test_parse_strict_2(self):
        self.assertEqual(_parse_pairwise_orient_response("2", "X", "Y"), ("Y", "X"))

    def test_parse_strict_a(self):
        self.assertEqual(_parse_pairwise_orient_response("A", "X", "Y"), ("X", "Y"))

    def test_parse_strict_b(self):
        self.assertEqual(_parse_pairwise_orient_response("b", "X", "Y"), ("Y", "X"))

    def test_parse_strict_asterisks(self):
        # Legacy behavior: '**1**' becomes '1' after replace('*','')
        self.assertEqual(_parse_pairwise_orient_response("**1**", "X", "Y"), ("X", "Y"))
        self.assertEqual(_parse_pairwise_orient_response("**2**", "X", "Y"), ("Y", "X"))

    def test_parse_tolerant_leading_punctuation(self):
        # "1." or " 2 " — local models often wrap the answer with punctuation.
        self.assertEqual(_parse_pairwise_orient_response("1.", "X", "Y"), ("X", "Y"))
        self.assertEqual(_parse_pairwise_orient_response(" 2 ", "X", "Y"), ("Y", "X"))
        self.assertEqual(_parse_pairwise_orient_response("- 1", "X", "Y"), ("X", "Y"))

    def test_parse_tolerant_followed_by_text(self):
        self.assertEqual(
            _parse_pairwise_orient_response("1. X causes Y because ...", "X", "Y"),
            ("X", "Y"),
        )
        self.assertEqual(
            _parse_pairwise_orient_response("2) Y -> X", "X", "Y"),
            ("Y", "X"),
        )

    def test_parse_unclear_raises(self):
        with self.assertRaises(ValueError) as cm:
            _parse_pairwise_orient_response("I am not sure about this", "X", "Y")
        self.assertIn("unclear", str(cm.exception).lower())
        # Error message must include the raw response for debuggability
        self.assertIn("I am not sure", str(cm.exception))

    def test_parse_empty_raises(self):
        with self.assertRaises(ValueError) as cm:
            _parse_pairwise_orient_response("", "X", "Y")
        self.assertIn("empty", str(cm.exception).lower())

    def test_parse_none_raises(self):
        with self.assertRaises(ValueError):
            _parse_pairwise_orient_response(None, "X", "Y")

    def test_parse_rejects_ambiguous_leading_text(self):
        # If the first non-punctuation token isn't 1/2/a/b, we should NOT guess.
        with self.assertRaises(ValueError):
            _parse_pairwise_orient_response("The answer is 1", "X", "Y")

    # --- Callable backend ---

    def test_callable_backend_returns_xy(self):
        def fake(prompt, **_):
            # Prompt must contain both descriptions.
            self.assertIn("age of a person", prompt)
            self.assertIn("income of a person", prompt)
            return "1"

        self.assertEqual(
            llm_pairwise_orient("Age", "Income", self.descriptions, backend=fake),
            ("Age", "Income"),
        )

    def test_callable_backend_returns_yx(self):
        self.assertEqual(
            llm_pairwise_orient("Age", "Income", self.descriptions, backend=lambda p, **_: "2"),
            ("Income", "Age"),
        )

    def test_callable_backend_receives_backend_kwargs(self):
        received = {}

        def fake(prompt, **kwargs):
            received.update(kwargs)
            return "1"

        llm_pairwise_orient(
            "Age", "Income", self.descriptions,
            backend=fake,
            backend_kwargs={"foo": "bar", "count": 3},
        )
        self.assertEqual(received, {"foo": "bar", "count": 3})

    def test_callable_backend_top_level_kwargs_override_backend_kwargs(self):
        received = {}

        def fake(prompt, **kwargs):
            received.update(kwargs)
            return "1"

        llm_pairwise_orient(
            "Age", "Income", self.descriptions,
            backend=fake,
            backend_kwargs={"shared": "from_backend_kwargs", "b_only": 1},
            shared="from_kwargs",
            k_only=2,
        )
        self.assertEqual(
            received,
            {"shared": "from_kwargs", "b_only": 1, "k_only": 2},
        )

    def test_callable_backend_unclear_response_raises(self):
        with self.assertRaises(ValueError):
            llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend=lambda p, **_: "I don't know",
            )

    # --- Unknown backend ---

    def test_unknown_backend_string_raises(self):
        with self.assertRaises(ValueError) as cm:
            llm_pairwise_orient("Age", "Income", self.descriptions, backend="bogus")
        self.assertIn("bogus", str(cm.exception))
        self.assertIn("litellm", str(cm.exception))

    def test_non_string_non_callable_backend_raises(self):
        with self.assertRaises(ValueError):
            llm_pairwise_orient("Age", "Income", self.descriptions, backend=42)

    # --- Litellm backend (mocked) ---

    @staticmethod
    def _make_fake_litellm(response_content="1"):
        fake_resp = mock.MagicMock()
        fake_resp.choices = [mock.MagicMock()]
        fake_resp.choices[0].message.content = response_content
        fake_completion = mock.MagicMock(return_value=fake_resp)
        fake_module = mock.MagicMock(completion=fake_completion)
        return fake_module, fake_completion

    def test_litellm_backend_mocked_returns_xy(self):
        fake_module, fake_completion = self._make_fake_litellm("1")
        with mock.patch.dict(sys.modules, {"litellm": fake_module}):
            result = llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="litellm", llm_model="gemini/test",
            )
        self.assertEqual(result, ("Age", "Income"))
        fake_completion.assert_called_once()
        call_kwargs = fake_completion.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini/test")
        self.assertEqual(call_kwargs["messages"][0]["role"], "user")
        self.assertIn("age of a person", call_kwargs["messages"][0]["content"])

    def test_litellm_is_default_backend(self):
        fake_module, fake_completion = self._make_fake_litellm("2")
        with mock.patch.dict(sys.modules, {"litellm": fake_module}):
            result = llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                llm_model="gemini/test",
            )
        self.assertEqual(result, ("Income", "Age"))

    def test_litellm_forwards_extra_kwargs(self):
        """Backward-compat: historically kwargs at top level went to completion()."""
        fake_module, fake_completion = self._make_fake_litellm("1")
        with mock.patch.dict(sys.modules, {"litellm": fake_module}):
            llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                llm_model="gemini/test",
                temperature=0.2,
                custom_param="x",
            )
        call_kwargs = fake_completion.call_args.kwargs
        self.assertEqual(call_kwargs.get("temperature"), 0.2)
        self.assertEqual(call_kwargs.get("custom_param"), "x")

    def test_litellm_backend_kwargs_also_forwarded(self):
        fake_module, fake_completion = self._make_fake_litellm("1")
        with mock.patch.dict(sys.modules, {"litellm": fake_module}):
            llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="litellm", llm_model="gemini/test",
                backend_kwargs={"temperature": 0.5, "top_p": 0.9},
            )
        call_kwargs = fake_completion.call_args.kwargs
        self.assertEqual(call_kwargs.get("temperature"), 0.5)
        self.assertEqual(call_kwargs.get("top_p"), 0.9)

    @pytest.mark.skipif(
        _check_soft_dependencies("litellm", severity="none"),
        reason="litellm is installed; this test verifies behavior when it is absent",
    )
    def test_litellm_not_installed_raises_importerror(self):
        # sys.modules['litellm'] = None blocks the import without needing
        # to actually uninstall the package.
        with mock.patch.dict(sys.modules, {"litellm": None}):
            with self.assertRaises(ImportError) as cm:
                llm_pairwise_orient("Age", "Income", self.descriptions, backend="litellm")
            self.assertIn("litellm", str(cm.exception).lower())

    # --- Ollama backend (mocked) ---

    @staticmethod
    def _make_fake_http_response(json_body):
        fake = mock.MagicMock()
        fake.json.return_value = json_body
        fake.raise_for_status = mock.MagicMock()
        return fake

    def test_ollama_backend_mocked(self):
        import requests
        fake_resp = self._make_fake_http_response({"response": "2"})
        with mock.patch.object(requests, "post", return_value=fake_resp) as mp:
            result = llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="ollama", llm_model="llama3",
                backend_kwargs={
                    "host": "http://myhost:11434",
                    "options": {"temperature": 0.0},
                },
            )
        self.assertEqual(result, ("Income", "Age"))
        args, kwargs = mp.call_args
        self.assertEqual(args[0], "http://myhost:11434/api/generate")
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "llama3")
        self.assertEqual(payload["stream"], False)
        self.assertEqual(payload["options"], {"temperature": 0.0})
        self.assertIn("age of a person", payload["prompt"])
        # Reasonable timeout was passed
        self.assertIn("timeout", kwargs)

    def test_ollama_default_host(self):
        import requests
        fake_resp = self._make_fake_http_response({"response": "1"})
        with mock.patch.object(requests, "post", return_value=fake_resp) as mp:
            llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="ollama", llm_model="llama3",
            )
        self.assertIn("localhost:11434", mp.call_args.args[0])

    def test_ollama_raises_on_http_error(self):
        import requests
        fake_resp = mock.MagicMock()
        fake_resp.raise_for_status.side_effect = requests.HTTPError("500")
        with mock.patch.object(requests, "post", return_value=fake_resp):
            with self.assertRaises(requests.HTTPError):
                llm_pairwise_orient(
                    "Age", "Income", self.descriptions,
                    backend="ollama", llm_model="llama3",
                )

    # --- OpenAI-compatible backend (mocked) ---

    def test_openai_compatible_mocked(self):
        import requests
        fake_resp = self._make_fake_http_response(
            {"choices": [{"message": {"content": "1"}}]}
        )
        with mock.patch.object(requests, "post", return_value=fake_resp) as mp:
            result = llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="openai_compatible", llm_model="llama-3-8b",
                backend_kwargs={
                    "base_url": "http://localhost:8000/v1",
                    "api_key": "sk-fake",
                    "temperature": 0.1,
                },
            )
        self.assertEqual(result, ("Age", "Income"))
        args, kwargs = mp.call_args
        self.assertEqual(args[0], "http://localhost:8000/v1/chat/completions")
        self.assertEqual(kwargs["headers"].get("Authorization"), "Bearer sk-fake")
        self.assertEqual(kwargs["headers"].get("Content-Type"), "application/json")
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "llama-3-8b")
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(payload["messages"][0]["role"], "user")

    def test_openai_compatible_no_api_key_omits_auth_header(self):
        import requests
        fake_resp = self._make_fake_http_response(
            {"choices": [{"message": {"content": "1"}}]}
        )
        with mock.patch.object(requests, "post", return_value=fake_resp) as mp:
            llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="openai_compatible", llm_model="m",
                backend_kwargs={"base_url": "http://localhost:8000/v1"},
            )
        self.assertNotIn("Authorization", mp.call_args.kwargs["headers"])

    def test_openai_compatible_missing_base_url_raises(self):
        with self.assertRaises(TypeError):
            llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="openai_compatible",
            )

    # --- Transformers backend (mocked; real integration is optional) ---

    @staticmethod
    def _make_fake_transformers(generator_output):
        captured = {"pipeline_calls": 0, "gen_calls": 0, "pipeline_kwargs": {}, "gen_kwargs": {}}

        def fake_generator(prompt, **gen_kwargs):
            captured["gen_calls"] += 1
            captured["gen_kwargs"] = gen_kwargs
            return generator_output

        def fake_pipeline(task, model=None, **kwargs):
            assert task == "text-generation", task
            captured["pipeline_calls"] += 1
            captured["pipeline_kwargs"] = {"model": model, **kwargs}
            return fake_generator

        fake_module = types.ModuleType("transformers")
        fake_module.pipeline = fake_pipeline
        return fake_module, captured

    def test_transformers_backend_mocked(self):
        _transformers_pipeline_cache.clear()
        fake_mod, captured = self._make_fake_transformers([{"generated_text": "1"}])
        with mock.patch.dict(sys.modules, {"transformers": fake_mod}):
            result = llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="transformers", llm_model="fake-model",
            )
        self.assertEqual(result, ("Age", "Income"))
        self.assertEqual(captured["pipeline_calls"], 1)
        self.assertEqual(captured["pipeline_kwargs"]["model"], "fake-model")
        self.assertEqual(captured["gen_kwargs"].get("max_new_tokens"), 16)
        self.assertEqual(captured["gen_kwargs"].get("return_full_text"), False)
        self.assertEqual(captured["gen_kwargs"].get("do_sample"), False)

    def test_transformers_pipeline_is_cached(self):
        _transformers_pipeline_cache.clear()
        fake_mod, captured = self._make_fake_transformers([{"generated_text": "1"}])
        with mock.patch.dict(sys.modules, {"transformers": fake_mod}):
            llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="transformers", llm_model="same-model",
            )
            llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="transformers", llm_model="same-model",
            )
        self.assertEqual(captured["pipeline_calls"], 1)
        self.assertEqual(captured["gen_calls"], 2)

    def test_transformers_pipeline_kwargs_forwarded_and_cache_skipped(self):
        _transformers_pipeline_cache.clear()
        fake_mod, captured = self._make_fake_transformers([{"generated_text": "2"}])
        with mock.patch.dict(sys.modules, {"transformers": fake_mod}):
            result = llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="transformers", llm_model="custom-model",
                backend_kwargs={
                    "pipeline_kwargs": {"device": "cpu"},
                    "max_new_tokens": 32,
                },
            )
        self.assertEqual(result, ("Income", "Age"))
        self.assertEqual(captured["pipeline_kwargs"].get("device"), "cpu")
        self.assertEqual(captured["gen_kwargs"].get("max_new_tokens"), 32)
        # Cache must be skipped when pipeline_kwargs are customized.
        self.assertNotIn("custom-model", _transformers_pipeline_cache)

    def test_transformers_not_installed_raises_importerror(self):
        _transformers_pipeline_cache.clear()
        # Blocking import via sys.modules['transformers'] = None only works
        # if the module hasn't already been imported. Since the test env may
        # have imported transformers, we use a monkey-patched sys.modules that
        # pops the real one first.
        saved = sys.modules.pop("transformers", None)
        try:
            with mock.patch.dict(sys.modules, {"transformers": None}):
                with self.assertRaises(ImportError) as cm:
                    llm_pairwise_orient(
                        "Age", "Income", self.descriptions,
                        backend="transformers", llm_model="m",
                    )
                self.assertIn("transformers", str(cm.exception).lower())
        finally:
            if saved is not None:
                sys.modules["transformers"] = saved

    # --- Backward-compatibility smoke test ---

    def test_old_style_signature_still_works(self):
        """Callers that don't know about `backend` must continue to work."""
        fake_module, fake_completion = self._make_fake_litellm("1")
        with mock.patch.dict(sys.modules, {"litellm": fake_module}):
            # This is the exact call shape used by the existing ExpertInLoop
            # docstring example and user code before this PR.
            result = llm_pairwise_orient(
                x="Age",
                y="Income",
                descriptions=self.descriptions,
                llm_model="gemini/gemini-1.5-flash",
            )
        self.assertEqual(result, ("Age", "Income"))

    def test_backend_kwarg_is_keyword_only(self):
        """`backend` must not accept a positional 6th argument, for safety."""
        with self.assertRaises(TypeError):
            llm_pairwise_orient("Age", "Income", self.descriptions, None,
                                "gemini/test", "litellm")

    # --- Optional integration test for transformers ---

    @pytest.mark.skipif(
        not _check_soft_dependencies("transformers", severity="none")
        or os.environ.get("PGMPY_RUN_TRANSFORMERS_INTEGRATION") != "1",
        reason="Opt-in: set PGMPY_RUN_TRANSFORMERS_INTEGRATION=1 to run the "
        "transformers integration test (downloads a small model).",
    )
    def test_transformers_backend_integration_tiny_model(self):
        _transformers_pipeline_cache.clear()
        # Use a tiny random model so the test is fast and doesn't need network
        # once cached. We can't assert a direction (the model is untrained),
        # so we only assert that the function either returns a valid tuple or
        # raises the expected ValueError on unclear output.
        try:
            result = llm_pairwise_orient(
                "Age", "Income", self.descriptions,
                backend="transformers",
                llm_model="sshleifer/tiny-gpt2",
            )
        except ValueError as e:
            self.assertIn("unclear", str(e).lower())
            return
        self.assertIn(result, [("Age", "Income"), ("Income", "Age")])


class TestPreprocessData(unittest.TestCase):
    def setUp(self):
        self.data_raw = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)

        self.data_proc = self.data_raw.copy()
        self.data_proc["A_cat"] = self.data_proc.A_cat.astype("category")
        self.data_proc["B_cat"] = self.data_proc.C_cat.astype("category")
        self.data_proc["C_cat"] = self.data_proc.C_cat.astype("category")

        self.data_proc_proc = self.data_proc.copy()
        cat_type = pd.CategoricalDtype(
            categories=np.array(sorted(self.data_proc_proc.B_int.unique())),
            ordered=True,
        )

        self.data_proc_proc["B_int"] = self.data_proc_proc.B_int.astype(cat_type)

    def test_preprocess_data(self):
        df, dtypes = preprocess_data(self.data_raw)
        self.assertEqual(
            dtypes,
            {
                "A": "N",
                "B": "N",
                "C": "N",
                "A_cat": "C",
                "B_cat": "C",
                "C_cat": "C",
                "B_int": "N",
            },
        )

        df, dtypes = preprocess_data(self.data_proc)
        self.assertEqual(
            dtypes,
            {
                "A": "N",
                "B": "N",
                "C": "N",
                "A_cat": "C",
                "B_cat": "C",
                "C_cat": "C",
                "B_int": "N",
            },
        )

        df, dtypes = preprocess_data(self.data_proc_proc)
        self.assertEqual(
            dtypes,
            {
                "A": "N",
                "B": "N",
                "C": "N",
                "A_cat": "C",
                "B_cat": "C",
                "C_cat": "C",
                "B_int": "O",
            },
        )


class TestGetExampleModel(unittest.TestCase):
    def test_get_categorical_models(self):
        """Test loading of categorical Bayesian network models."""
        cat_models = {
            "asia",
            "cancer",
            "earthquake",
            "sachs",
            "survey",
            "alarm",
            "barley",
            "child",
            "insurance",
            "mildew",
            "water",
            "hailfinder",
            "hepar2",
            "win95pts",
            "andes",
            "diabetes",
            "link",
            "munin1",
            "munin2",
            "munin3",
            "munin4",
            "pathfinder",
            "pigs",
            "munin",
        }

        # Randomly select 5 categorical models to test
        choices = random.sample(list(cat_models), k=5)
        for model in tqdm(choices, desc="Testing categorical models"):
            m = get_example_model(model=model)
            # Basic model validation
            self.assertIsNotNone(m)
            self.assertTrue(hasattr(m, "nodes"))
            self.assertTrue(hasattr(m, "edges"))
            del m

    def test_get_continuous_models(self):
        # Test ecoli70 model specifically as we have its structure
        model = get_example_model("ecoli70")
        self.assertIsInstance(model, LinearGaussianBayesianNetwork)
        self.assertEqual(len(model.nodes()), 46)  # Number of nodes in ecoli70

        # Verify some known relationships from the provided structure
        self.assertIn(("asnA", "icdA"), model.edges())
        self.assertIn(("asnA", "lacA"), model.edges())
        self.assertIn(("sucA", "atpD"), model.edges())

        # Verify CPD structure for a known node
        cpd = model.get_cpds("aceB")
        self.assertIsNotNone(cpd)
        self.assertEqual(cpd.variable, "aceB")
        self.assertEqual(len(cpd.evidence), 1)
        self.assertIn("icdA", cpd.evidence)

    def test_get_example_model_dagitty(self):
        dag_models = [
            "M-bias",
            "confounding",
            "mediator",
            "paths",
            "Sebastiani_2005",
            "Polzer_2012",
            "Schipf_2010",
            "Shrier_2008",
            "Acid_1996",
            "Thoemmes_2013",
            "Kampen_2014",
            "Didelez_2010",
        ]
        # Would take too much time to load all the models. Hence, randomly select
        # 3 and try to load them.
        choices = random.sample(dag_models, k=3)
        for model in tqdm(choices):
            print(model)
            m = get_example_model(model=model)
            self.assertIsNotNone(m)
            self.assertTrue(hasattr(m, "nodes"))
            self.assertTrue(hasattr(m, "edges"))
            del m

    def test_invalid_model_name(self):
        """Test handling of invalid model names."""
        with self.assertRaises(ValueError):
            get_example_model("nonexistent_model")

    def test_model_categorization(self):
        """Test that all models are properly categorized."""
        # Test a model from each category
        cat_model = get_example_model("asia")
        self.assertNotIsInstance(cat_model, LinearGaussianBayesianNetwork)

        cont_model = get_example_model("magic-irri")
        self.assertIsInstance(cont_model, LinearGaussianBayesianNetwork)
