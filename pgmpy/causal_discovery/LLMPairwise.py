from __future__ import annotations

import re

import networkx as nx
import pandas as pd
from skbase.utils.dependencies import _safe_import

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery

litellm = _safe_import("litellm")


class LLMPairwise(BaseCausalDiscovery):
    """
    LLM-based pairwise causal discovery estimator.

    Orients the edge between exactly two variables by querying a Large Language Model with the variable names and
    optional text descriptions. The data values themselves are not used; they only provide the standard pgmpy estimator
    ``fit`` interface.

    Parameters
    ----------
    descriptions : dict, default=None
        Mapping from variable names to text descriptions. If a variable is missing, its name is used as the description.

    system_prompt : str, default=None
        A system prompt to give the LLM. If ``None``, defaults to ``"You are an expert in Causal Inference"``.

    llm_model : str, default="gemini/gemini-1.5-flash"
        The LLM model to use. This can be any model supported by LiteLLM. Please refer to the LiteLLM documentation
        (https://docs.litellm.ai/docs/providers) for the full list.

    llm_kwargs : dict, default=None
        Additional keyword arguments passed to ``litellm.completion``, for example ``{"temperature": 0}``.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph as a DAG.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of ``causal_graph_``.

    llm_model_ : str
        The LLM model that was used to orient the edge.

    llm_prompt_ : list
        The chat messages sent to the LLM.

    llm_response_ : str
        The raw text response returned by the LLM.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Notes
    -----
    The query is sent through ``LiteLLM``, so the provider's API key must be set before calling ``fit`` (e.g. via the
    ``GEMINI_API_KEY`` or ``OPENAI_API_KEY`` environment variable). Locally served models (e.g. Ollama) need no API key
    but require the API endpoint to be specified in ``llm_kwargs``.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import LLMPairwise
    >>> df = pd.DataFrame({"Smoker": [0, 1], "Cancer": [0, 1]})
    >>> descriptions = {
    ...     "Smoker": "Whether a person smokes",
    ...     "Cancer": "Whether a person has cancer",
    ... }
    >>> est = LLMPairwise(descriptions=descriptions).fit(df)  # doctest: +SKIP
    >>> est.causal_graph_.edges()  # doctest: +SKIP
    OutEdgeView([('Smoker', 'Cancer')])

    A different hosted provider can be selected through ``llm_model``:

    >>> est = LLMPairwise(descriptions=descriptions, llm_model="openai/gpt-4o").fit(df)  # doctest: +SKIP

    A model served locally with Ollama can be used by prefixing the model name
    with ``ollama/`` and pointing ``api_base`` at the local server:

    >>> est = LLMPairwise(
    ...     descriptions=descriptions,
    ...     llm_model="ollama/llama3",
    ...     llm_kwargs={"api_base": "http://localhost:11434"},
    ... ).fit(df)  # doctest: +SKIP
    """

    def __init__(
        self,
        descriptions: dict | None = None,
        system_prompt: str | None = None,
        llm_model: str = "gemini/gemini-1.5-flash",
        llm_kwargs: dict | None = None,
    ):
        self.descriptions = descriptions
        self.system_prompt = system_prompt
        self.llm_model = llm_model
        self.llm_kwargs = llm_kwargs

    def _fit(self, X: pd.DataFrame):
        """
        Orient the edge between the two variables in `X` using an LLM.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from. Must contain exactly
            two variables.

        Returns
        -------
        self : LLMPairwise
            Returns the instance with the fitted attributes set.
        """
        # Step 1: This estimator only orients a single pair of variables.
        if X.shape[1] != 2:
            raise ValueError(f"LLMPairwise requires exactly two variables, got {X.shape[1]}.")

        # Step 2: Build the prompt from the variable names and descriptions.
        x, y = self.feature_names_in_
        self.llm_model_ = self.llm_model
        self.llm_prompt_ = self._build_prompt(x, y)

        # Step 3: Query the LLM and parse the chosen direction.
        self.llm_response_ = self._query_llm(self.llm_prompt_)
        source, target = self._parse_response(self.llm_response_, x, y)

        # Step 4: Build the causal graph and store the fitted attributes.
        dag = DAG([(source, target)])
        self.causal_graph_ = dag
        self.adjacency_matrix_ = nx.to_pandas_adjacency(dag, nodelist=[x, y], weight=None, dtype="int")

        return self

    def _build_prompt(self, x, y):
        """Build the system and user chat messages describing `x` and `y`."""
        descriptions = self.descriptions if self.descriptions is not None else {}
        if self.system_prompt is None:
            self.system_prompt = "You are an expert in Causal Inference"

        user_prompt = (
            "You are given two variables with the following descriptions:\n"
            f"<A>: {descriptions.get(x, x)}\n"
            f"<B>: {descriptions.get(y, y)}\n\n"
            "Which of the following two options is the most likely causal direction between them:\n"
            "1. <A> causes <B>\n"
            "2. <B> causes <A>\n\n"
            "Return a single number (1 or 2) as your answer. I do not need the reasoning behind it.\n"
            "Do not add any formatting in the answer."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _query_llm(self, messages):
        """Send `messages` to the LLM and return its text response."""
        llm_kwargs = self.llm_kwargs if self.llm_kwargs is not None else {}
        response = litellm.completion(model=self.llm_model, messages=messages, **llm_kwargs)
        return response.choices[0].message.content

    def _parse_response(self, response, x, y):
        """Parse the LLM `response` into a directed (source, target) edge."""
        response_txt = response.strip().lower().replace("*", "")

        # An option number takes precedence over a letter (e.g. "Option 1", "Answer: 2").
        number = re.search(r"[12]", response_txt)
        if number is not None:
            return (x, y) if number.group() == "1" else (y, x)

        letter = re.search(r"\b[ab]\b", response_txt)
        if letter is not None:
            return (x, y) if letter.group() == "a" else (y, x)

        raise ValueError("Results from the LLM are unclear. Try calling the estimator again.")
