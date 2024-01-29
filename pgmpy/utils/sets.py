from collections.abc import Iterable
from itertools import combinations, chain


def _variable_or_iterable_to_set(x):
    """
    Convert variable, set, or iterable x to a frozenset.

    If x is None, returns the empty set.

    Parameters
    ---------
    x : None, str or Iterable[str]

    Returns
    -------
    frozenset : frozenset representation of string or iterable input
    """
    if x is None:
        return frozenset([])

    if isinstance(x, str):
        return frozenset([x])

    if not isinstance(x, Iterable) or not all(isinstance(xx, str) for xx in x):
        raise ValueError(
            f"{x} is expected to be either a string, set of strings, or an iterable of strings"
        )

    return frozenset(x)


def _powerset(iterable):
    """
    https://docs.python.org/3/library/itertools.html#recipes
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Parameters
    ----------
    iterable: any iterable

    Returns
    -------
    chain: a generator of the powerset of the input
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
