#!/usr/bin/env python
from pgmpy.estimators import StructureScore


class ScoreCache(StructureScore):
    """
    A wrapper class for StructureScore instances, which implement a decomposable score,
    that caches local scores.
    Based on the global decomposition property of Bayesian networks for decomposable scores.

    Parameters
    ----------
    base_scorer: StructureScore instance
         Has to be a decomposable score.
    data: pandas DataFrame instance
        DataFrame instance where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)
    max_size: int (optional, default 10_000)
        The maximum number of elements allowed in the cache. When the limit is reached, the least recently used
        entries will be discarded.
    **kwargs
        Additional arguments that will be handed to the super constructor.

    Reference
    ---------
    Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3
    """

    def __init__(self, base_scorer, data, max_size=10000, **kwargs):
        assert isinstance(
            base_scorer, StructureScore
        ), "Base scorer has to be of type StructureScore."

        self.base_scorer = base_scorer
        self.cache = LRUCache(
            original_function=self._wrapped_original, max_size=int(max_size)
        )
        super(ScoreCache, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        hashable = tuple(parents)
        return self.cache(variable, hashable)

    def _wrapped_original(self, variable, parents):
        expected = list(parents)
        return self.base_scorer.local_score(variable, expected)


# link fields
_PREV, _NEXT, _KEY, _VALUE = 0, 1, 2, 3


class LRUCache:
    """
    Least-Recently-Used cache.
    Acts as a wrapper around an arbitrary function and caches the return values.

    Based on the implementation of Raymond Hettinger
    (https://stackoverflow.com/questions/2437617/limiting-the-size-of-a-python-dictionary)

    Parameters
    ----------
    original_function: callable
        The original function that will be wrapped. Return values will be cached.
        The function parameters have to be hashable.
    max_size: int (optional, default 10_000)
        The maximum number of elements allowed within the cache. If the size would be exceeded,
        the least recently used element will be removed from the cache.
    """

    def __init__(self, original_function, max_size=10000):
        self.original_function = original_function
        self.max_size = max_size
        self.mapping = {}

        # oldest
        self.head = [None, None, None, None]
        # newest
        self.tail = [self.head, None, None, None]
        self.head[_NEXT] = self.tail

    def __call__(self, *key):
        mapping, head, tail = self.mapping, self.head, self.tail

        link = mapping.get(key, head)
        if link is head:
            # Not yet in map
            value = self.original_function(*key)
            if len(mapping) >= self.max_size:
                # Unlink the least recently used element
                old_prev, old_next, old_key, old_value = head[_NEXT]
                head[_NEXT] = old_next
                old_next[_PREV] = head
                del mapping[old_key]
            # Add new value as most recently used element
            last = tail[_PREV]
            link = [last, tail, key, value]
            mapping[key] = last[_NEXT] = tail[_PREV] = link
        else:
            # Unlink element from current position
            link_prev, link_next, key, value = link
            link_prev[_NEXT] = link_next
            link_next[_PREV] = link_prev
            # Add as most recently used element
            last = tail[_PREV]
            last[_NEXT] = tail[_PREV] = link
            link[_PREV] = last
            link[_NEXT] = tail
        return value
