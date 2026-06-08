from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery


class LLMPairwise(BaseCausalDiscovery):
    """
    Pairwise causal discovery estimator using a Large Language Model.

    The estimator asks an LLM to orient the edge between exactly two variables
    using their names and optional text descriptions. The data values are not
    used for scoring; they are only used to provide the standard pgmpy
    estimator ``fit`` interface.

    Parameters
    ----------
    descriptions : dict, optional
        Mapping from variable names to text descriptions. If a variable is
        missing, its name is used as the description.

    system_prompt : str, optional
        System instruction prepended to the prompt. If ``None``, defaults to
        ``"You are an expert in Causal Inference"``.

    llm_model : str, default="gemini/gemini-1.5-flash"
        The model name passed to ``litellm.completion``.

    llm_kwargs : dict, optional
        Additional keyword arguments passed to ``litellm.completion``.

    show_progress : bool, default=True
        Kept for consistency with other causal discovery estimators.

    Attributes
    ----------
    causal_graph_ : DAG
        The learned two-node causal graph.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of ``causal_graph_``.

    direction_score_ : float
        ``1.0`` when the first input column is oriented toward the second input
        column, and ``-1.0`` for the reverse direction. This is not a calibrated
        confidence score.

    prompt_ : str
        Prompt sent to the LLM.

    response_ : str
        Raw text response returned by the LLM.

    n_features_in_ : int
        The number of features in the data used to fit the estimator.

    feature_names_in_ : np.ndarray
        The feature names in the data used to fit the estimator.

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
        descriptions: dict[Hashable, str] | None = None,
        system_prompt: str | None = None,
        llm_model: str = "gemini/gemini-1.5-flash",
        llm_kwargs: dict | None = None,
        show_progress: bool = True,
    ):
        self.descriptions = descriptions
        self.system_prompt = system_prompt
        self.llm_model = llm_model
        self.llm_kwargs = llm_kwargs
        self.show_progress = show_progress

    def _fit(self, X: pd.DataFrame):
        """
        Fit the estimator on a two-column dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset containing exactly two variables.

        Returns
        -------
        self : LLMPairwise
            Fitted estimator.
        """
        if X.shape[1] != 2:
            raise ValueError("LLMPairwise requires exactly two variables.")

        x, y = X.columns
        self.variables_ = [x, y]
        self.prompt_ = self._build_prompt(x, y)
        response = self._call_llm(self.prompt_)
        self.response_ = response.choices[0].message.content

        edge = self._parse_response(self.response_, x, y)
        self.direction_score_ = 1.0 if edge == (x, y) else -1.0

        dag = DAG()
        dag.add_nodes_from(self.variables_)
        dag.add_edge(*edge)

        self.causal_graph_ = dag
        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            dag,
            nodelist=self.variables_,
            weight=None,
            dtype="int",
        )

        return self

    def _build_prompt(self, x: Hashable, y: Hashable) -> str:
        descriptions = self.descriptions or {}
        system_prompt = self.system_prompt
        if system_prompt is None:
            system_prompt = "You are an expert in Causal Inference"

        x_description = descriptions.get(x, str(x))
        y_description = descriptions.get(y, str(y))

        return f""" {system_prompt}. You are
      given two variables with the following descriptions:
        <A>: {x_description}
        <B>: {y_description}

        Which of the following two options is the most likely causal direction between them:
        1. <A> causes <B>
        2. <B> causes <A>

        Return a single number (1 or 2) as your answer. I do not need the reasoning behind it.
        Do not add any formatting in the answer.
        """

    def _call_llm(self, prompt: str):
        try:
            from litellm import completion
        except ImportError as e:
            raise ImportError(
                f"{e}. litellm is required for using"
                " LLM based pairwise orientation. "
                "Please install using: pip install litellm"
            ) from None

        llm_kwargs = self.llm_kwargs or {}
        return completion(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            **llm_kwargs,
        )

    def _parse_response(self, response: str, x: Hashable, y: Hashable) -> tuple[Hashable, Hashable]:
        response_txt = response.strip().lower().replace("*", "")
        if response_txt in ("a", "1"):
            return (x, y)
        elif response_txt in ("b", "2"):
            return (y, x)
        else:
            raise ValueError("Results from the LLM are unclear. Try calling the estimator again.")
