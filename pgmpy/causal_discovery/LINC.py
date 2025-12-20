from typing import Optional, Union

import pandas as pd

from pgmpy.causal_discovery.TOPIC import TOPIC
from pgmpy.estimators import ExpertKnowledge, StructureScore
from pgmpy.estimators.ScoreCache import ScoreCache
from pgmpy.estimators.StructureScore import get_scoring_method


class LINC(TOPIC):
    """
    LINC: TOPIC-style causal discovery for interventional / multi-context data.

    Expects one column in X to represent the context variable. Scoring is done
    independently per context and summed.
    """

    def __init__(
        self,
        context_col: str,
        variant: str = "parallel",
        return_type: str = "dag",
        scoring_method: Optional[Union[str, StructureScore]] = None,
        significance_level: float = 0.05,
        expert_knowledge: Optional[ExpertKnowledge] = None,
        enforce_expert_knowledge: bool = False,
        n_jobs: int = -1,
        show_progress: bool = True,
        use_cache: bool = True,
    ):
        super().__init__(
            variant=variant,
            return_type=return_type,
            scoring_method=scoring_method,
            significance_level=significance_level,
            expert_knowledge=expert_knowledge,
            enforce_expert_knowledge=enforce_expert_knowledge,
            n_jobs=n_jobs,
            show_progress=show_progress,
            use_cache=use_cache,
        )
        self.context_col = context_col
        self.context_data = None
        self.context_score_fns = None

    def _init_score(self, X: pd.DataFrame):
        if self.context_col not in X.columns:
            raise ValueError(f"context_col '{self.context_col}' not found in X.columns")

        # Split contexts
        contexts = X[self.context_col].unique()
        self.context_data = {}
        self.context_score_fns = {}
        self.score = None

        for c in contexts:
            Xc = X[X[self.context_col] == c].drop(columns=[self.context_col])

            score_c: ScoreCache
            score, score_c = get_scoring_method(self.scoring_method, Xc, self.use_cache)

            self.context_data[c] = Xc
            self.context_score_fns[c] = score_c.local_score

            if self.score is None:
                self.score = score

    def _score(self, effect, parents) -> float:
        total = 0.0
        for score_fn in self.context_score_fns.values():
            total += score_fn(effect, parents)
        return total

    def fit(self, X: pd.DataFrame, **kwargs):
        if self.context_col not in X.columns:
            raise ValueError(f"context_col '{self.context_col}' not found in X.columns")

        self._init_score(X)

        X_data = X.drop(columns=[self.context_col]).copy()

        orig_init_score = self._init_score
        try:
            self._init_score = lambda _X: None
            return super().fit(X_data, **kwargs)
        finally:
            self._init_score = orig_init_score
