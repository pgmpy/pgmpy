from dataclasses import dataclass
from typing import Callable, FrozenSet, Hashable, Iterable, Literal

import numpy as np
import pandas as pd

from pgmpy.causal_discovery.TOPIC import TOPIC
from pgmpy.estimators.StructureScore import BICGauss, get_scoring_method

ScoreFn = Callable[[str, tuple[str, ...]], float]


@dataclass(frozen=True)
class _GroupingParams:
    gain_threshold: float = 0.0
    maximize: bool = True
    method: Literal["components", "agglomerative"] = "components"


def _union_find_components(
    nodes: list[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
) -> list[list[Hashable]]:
    parent = {x: x for x in nodes}
    rank = {x: 0 for x in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in edges:
        union(a, b)

    comps: dict[Hashable, list[Hashable]] = {}
    for x in nodes:
        rx = find(x)
        comps.setdefault(rx, []).append(x)
    return list(comps.values())


class LINC(TOPIC):
    score_ = BICGauss
    score_fn_ = None

    def __init__(
        self,
        *args,
        context_col: str = "context",
        grouping: _GroupingParams | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.context_col = context_col

        self._X_context: dict[Hashable, pd.DataFrame] = {}
        self.context_score_fns: dict[Hashable, ScoreFn] = {}

        self.grouping = grouping or _GroupingParams()
        self._pooled_score_fns: dict[FrozenSet[Hashable], ScoreFn] = {}

        self._last_gain_matrix: np.ndarray | None = None
        self._last_gain_contexts: tuple[Hashable, ...] | None = None

    def _parents_key(self, parents):
        return tuple(sorted(map(str, parents)))

    def _init_score(self, X: pd.DataFrame):
        if self.context_col not in X.columns:
            raise ValueError(f"context_col '{self.context_col}' not found in X.columns")

        self._X_context = {}
        for context, g in X.groupby(self.context_col):
            self._X_context[context] = g.drop(columns=[self.context_col]).copy()

        self.context_score_fns = {}
        for context, Xc in self._X_context.items():
            _, score_cache = get_scoring_method(self.scoring_method, Xc, True)
            self.context_score_fns[context] = score_cache.local_score

        self._pooled_score_fns = {}

    def _score_simple(self, effect, parents) -> float:
        # simple baseline for comparison, to see what happens if we simply sum scores across contexts.
        parents_t = tuple(parents) if parents is not None else tuple()
        return float(
            sum(fn(effect, parents_t) for fn in self.context_score_fns.values())
        )

    def _make_score_fn_for_df(self, df: pd.DataFrame) -> ScoreFn:
        _, score_cache = get_scoring_method(self.scoring_method, df, True)
        return score_cache.local_score

    def _get_group_score_fn(self, contexts: FrozenSet[Hashable]) -> ScoreFn:
        fn = self._pooled_score_fns.get(contexts)
        if fn is not None:
            return fn

        pooled = pd.concat(
            [self._X_context[c] for c in contexts], axis=0, ignore_index=True
        )
        fn = self._make_score_fn_for_df(pooled)
        self._pooled_score_fns[contexts] = fn
        return fn

    def _score_components(self, effect, parents) -> float:
        parents_t = tuple(parents) if parents is not None else tuple()

        ctx_ids = list(self.context_score_fns.keys())
        n = len(ctx_ids)
        if n == 0:
            return 0.0
        if n == 1:
            c = ctx_ids[0]
            return float(self.context_score_fns[c](effect, parents_t))

        s_ctx = {
            c: float(self.context_score_fns[c](effect, parents_t)) for c in ctx_ids
        }

        edges: list[tuple[Hashable, Hashable]] = []
        gain = np.zeros((n, n), dtype=float)

        for a_i in range(n):
            for b_i in range(a_i + 1, n):
                ca, cb = ctx_ids[a_i], ctx_ids[b_i]
                pair = frozenset((ca, cb))

                pooled_fn = self._get_group_score_fn(pair)
                s_pooled = float(pooled_fn(effect, parents_t))

                if self.grouping.maximize:
                    g = s_pooled - (s_ctx[ca] + s_ctx[cb])
                else:
                    g = (s_ctx[ca] + s_ctx[cb]) - s_pooled

                if g > self.grouping.gain_threshold:
                    edges.append((ca, cb))

                gain[a_i, b_i] = gain[b_i, a_i] = g

        self._last_gain_matrix = gain
        self._last_gain_contexts = tuple(ctx_ids)

        components = _union_find_components(ctx_ids, edges)

        total = 0.0
        for comp in components:
            comp_set = frozenset(comp)
            group_fn = self._get_group_score_fn(comp_set)
            total += float(group_fn(effect, parents_t))

        return float(total)

    def _score_agglomerative(self, effect, parents) -> float:
        parents_t = tuple(parents) if parents is not None else tuple()

        ctx_ids = list(self.context_score_fns.keys())
        if not ctx_ids:
            return 0.0
        if len(ctx_ids) == 1:
            c = ctx_ids[0]
            return float(self.context_score_fns[c](effect, parents_t))

        # start with singleton groups
        groups: list[FrozenSet[Hashable]] = [frozenset([c]) for c in ctx_ids]

        def group_score(g: FrozenSet[Hashable]) -> float:
            if len(g) == 1:
                (c,) = tuple(g)
                return float(self.context_score_fns[c](effect, parents_t))
            return float(self._get_group_score_fn(g)(effect, parents_t))

        # compute current group scores
        scores = {g: group_score(g) for g in groups}

        while True:
            best_gain = None
            best_pair = None
            best_merged = None

            # find best merge
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    gi, gj = groups[i], groups[j]
                    merged = gi | gj

                    s_merged = group_score(merged)

                    if self.grouping.maximize:
                        g = s_merged - (scores[gi] + scores[gj])
                    else:
                        g = (scores[gi] + scores[gj]) - s_merged

                    if best_gain is None or g > best_gain:
                        best_gain = g
                        best_pair = (gi, gj)
                        best_merged = merged

            if best_gain is None or best_gain <= self.grouping.gain_threshold:
                break

            gi, gj = best_pair  # type: ignore[misc]
            merged = best_merged  # type: ignore[misc]

            # update groups: remove gi, gj; add merged
            groups = [g for g in groups if g not in (gi, gj)]
            groups.append(merged)

            # update scores
            scores.pop(gi, None)
            scores.pop(gj, None)
            scores[merged] = group_score(merged)

        # final score = sum pooled score over groups
        return float(sum(scores[g] for g in groups))

    def _score(self, effect, parents) -> float:
        if self.grouping.method == "agglomerative":
            return self._score_agglomerative(effect, parents)
        elif self.grouping.method == "components":
            return self._score_components(effect, parents)
        else:
            assert self.grouping.method == "_baseline_sum"
            return self._score_simple(effect, parents)

    def fit(self, X: pd.DataFrame, **kwargs):
        if self.context_col not in X.columns:
            raise ValueError(f"context_col '{self.context_col}' not found in X.columns")

        self._init_score(X)
        X_data = X.drop(columns=[self.context_col]).copy()

        orig_init = self._init_score
        try:
            self._init_score = lambda _X: None
            return super().fit(X_data, **kwargs)
        finally:
            self._init_score = orig_init
