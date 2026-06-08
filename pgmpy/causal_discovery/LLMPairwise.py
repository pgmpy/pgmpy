from __future__ import annotations

import re

import networkx as nx
import pandas as pd
from skbase.utils.dependencies import _safe_import

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery


class LLMPairwise(BaseCausalDiscovery):
    """
    LLM-based pairwise causal discovery estimator.

    Orients the edge between exactly two variables by querying a Large Language
    Model with the variable names and optional text descriptions. The data
    values themselves are not used; they only provide the standard pgmpy
    estimator ``fit`` interface.

    Parameters
    ----------
    descriptions : dict, default=None
        Mapping from variable names to text descriptions. If a variable is
        missing, its name is used as the description.

    system_prompt : str, default=None
        A system prompt to give the LLM. If ``None``, defaults to
        ``"You are an expert in Causal Inference"``.

    llm_model : str, default="gemini/gemini-1.5-flash"
        The LLM model to use. Please refer to the litellm documentation
        (https://docs.litellm.ai/docs/providers) for the available models.

    llm_kwargs : dict, default=None
        Additional keyword arguments passed to ``litellm.completion``, for
        example ``{"temperature": 0}``.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph as a DAG.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of ``causal_graph_``.

    direction_score_ : float
        Orientation indicator (not a calibrated confidence). It is ``1.0`` when
        the edge points from the first variable to the second and ``-1.0`` when
        it points the other way.

    prompt_ : list
        The chat messages sent to the LLM.

    response_ : str
        The raw text response returned by the LLM.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import LLMPairwise
    >>> data = pd.DataFrame({"Smoker": [0, 1], "Cancer": [0, 1]})
    >>> descriptions = {
    ...     "Smoker": "Whether a person smokes",
    ...     "Cancer": "Whether a person has cancer",
    ... }
    >>> est = LLMPairwise(descriptions=descriptions).fit(data)  # doctest: +SKIP
    >>> est.causal_graph_.edges()  # doctest: +SKIP
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
        x, y = X.columns
        self.prompt_ = self._build_prompt(x, y)

        # Step 3: Query the LLM and parse the chosen direction.
        self.response_ = self._query_llm(self.prompt_)
        source, target = self._parse_response(self.response_, x, y)
        self.direction_score_ = 1.0 if (source, target) == (x, y) else -1.0

        # Step 4: Build the causal graph and store the fitted attributes.
        dag = DAG([(source, target)])
        self.causal_graph_ = dag
        self.adjacency_matrix_ = nx.to_pandas_adjacency(dag, nodelist=[x, y], weight=None, dtype="int")

        return self

    def _build_prompt(self, x, y):
        """Build the system and user chat messages describing `x` and `y`."""
        descriptions = self.descriptions if self.descriptions is not None else {}
        system_prompt = self.system_prompt
        if system_prompt is None:
            system_prompt = "You are an expert in Causal Inference"

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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _query_llm(self, messages):
        """Send `messages` to the LLM and return its text response."""
        litellm = _safe_import("litellm")

        llm_kwargs = self.llm_kwargs if self.llm_kwargs is not None else {}
        response = litellm.completion(model=self.llm_model, messages=messages, **llm_kwargs)
        return response.choices[0].message.content

    def _parse_response(self, response, x, y):
        """Parse the LLM `response` into a directed (source, target) edge."""
        response_txt = response.strip().lower().replace("*", "")

        # An explicit option number takes precedence over an option letter, so
        # that responses like "1.", "Option 1" or "Answer: 2" are parsed.
        number = re.search(r"[12]", response_txt)
        if number is not None:
            return (x, y) if number.group() == "1" else (y, x)

        letter = re.search(r"\b[ab]\b", response_txt)
        if letter is not None:
            return (x, y) if letter.group() == "a" else (y, x)

        raise ValueError("Results from the LLM are unclear. Try calling the estimator again.")
