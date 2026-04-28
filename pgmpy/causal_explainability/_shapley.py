from itertools import permutations
from math import factorial

import numpy as np


class ShapleyEngine:
    """Computes Shapley values given a value function.

    Parameters
    ----------
    n_players : int
        The number of players in the cooperative game.
    value_function : callable(coalition: frozenset) -> float
        A function that takes a frozenset of player indices and returns
        the value of that coalition.
    causal_ordering : list of sets, optional
        Causal ordering as tiers. When provided, uses asymmetric Shapley
        values that only average over causally consistent permutations.
    method : str
        "exact", "sampling", or "auto" (exact if n_players <= 15).
    """

    _AUTO_THRESHOLD = 15

    def __init__(self, n_players, value_function, causal_ordering=None, method="auto"):
        self.n_players = n_players
        self.value_function = value_function
        self.causal_ordering = causal_ordering
        self._cache = {}

        if method == "auto":
            self._resolved_method = "exact" if n_players <= self._AUTO_THRESHOLD else "sampling"
        else:
            self._resolved_method = method

    def _cached_value(self, coalition):
        key = coalition
        if key not in self._cache:
            self._cache[key] = self.value_function(coalition)
        return self._cache[key]

    def compute(self, n_permutations=1000, seed=None):
        """Compute Shapley values for all players.

        Parameters
        ----------
        n_permutations : int
            Number of permutations for sampling method. Ignored for exact.
        seed : int, optional
            Random seed for sampling method.

        Returns
        -------
        dict
            Mapping of player index to Shapley value.
        """
        if self._resolved_method == "exact":
            if self.causal_ordering is not None:
                return self._exact_asymmetric()
            return self._exact_symmetric()
        return self._sampling(n_permutations, seed)

    def _exact_symmetric(self):
        n = self.n_players
        players = range(n)
        shapley = dict.fromkeys(players, 0.0)
        for i in players:
            for size in range(0, n):
                others = [j for j in players if j != i]
                for combo in _combinations(others, size):
                    s = frozenset(combo)
                    s_with_i = s | {i}
                    marginal = self._cached_value(s_with_i) - self._cached_value(s)
                    weight = factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
                    shapley[i] += weight * marginal
        return shapley

    def _exact_asymmetric(self):
        n = self.n_players
        valid_perms = list(_causal_consistent_permutations(list(range(n)), self.causal_ordering))
        n_perms = len(valid_perms)
        shapley = dict.fromkeys(range(n), 0.0)
        for perm in valid_perms:
            predecessors = frozenset()
            for player in perm:
                with_player = predecessors | {player}
                marginal = self._cached_value(with_player) - self._cached_value(predecessors)
                shapley[player] += marginal / n_perms
                predecessors = with_player
        return shapley

    def _sampling(self, n_permutations, seed):
        rng = np.random.default_rng(seed)
        n = self.n_players
        shapley = dict.fromkeys(range(n), 0.0)
        players = list(range(n))
        for _ in range(n_permutations):
            if self.causal_ordering is not None:
                perm = _sample_causal_permutation(self.causal_ordering, rng)
            else:
                perm = list(rng.permutation(players))
            predecessors = frozenset()
            for player in perm:
                with_player = predecessors | {player}
                marginal = self._cached_value(with_player) - self._cached_value(predecessors)
                shapley[player] += marginal
                predecessors = with_player
        for i in range(n):
            shapley[i] /= n_permutations
        return shapley


def _combinations(items, r):
    """Generate all combinations of `r` items from `items`."""
    if r == 0:
        yield ()
        return
    for i in range(len(items)):
        for rest in _combinations(items[i + 1 :], r - 1):
            yield (items[i],) + rest


def _causal_consistent_permutations(players, ordering):
    """Generate all permutations consistent with a causal ordering.

    Parameters
    ----------
    players : list
        List of player indices.
    ordering : list of sets
        Causal ordering as tiers. Players in tier i must appear
        before players in tier i+1.

    Yields
    ------
    tuple
        A permutation consistent with the causal ordering.
    """
    tier_players = [sorted(tier) for tier in ordering]

    def _gen(tier_idx, used):
        if tier_idx == len(tier_players):
            yield ()
            return
        tier = [p for p in tier_players[tier_idx] if p not in used]
        for tier_perm in permutations(tier):
            for rest in _gen(tier_idx + 1, used | set(tier_perm)):
                yield tier_perm + rest

    yield from _gen(0, set())


def _sample_causal_permutation(ordering, rng):
    """Sample a single permutation consistent with a causal ordering.

    Parameters
    ----------
    ordering : list of sets
        Causal ordering as tiers.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    list
        A permutation consistent with the causal ordering.
    """
    perm = []
    for tier in ordering:
        tier_list = sorted(tier)
        rng.shuffle(tier_list)
        perm.extend(tier_list)
    return perm
