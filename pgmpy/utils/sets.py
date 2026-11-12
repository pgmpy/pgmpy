from collections.abc import Iterable
from itertools import chain, combinations


def _variable_or_iterable_to_set(x):
    """
    Convert variable, set, or iterable x to a frozenset.

    If x is None, returns the empty set. A string is one variable.

    Parameters
    ---------
    x : None, str or Iterable[Hashable]

    Returns
    -------
    frozenset : frozenset representation of string or iterable input
    """
    if x is None:
        return frozenset([])

    if isinstance(x, str):
        return frozenset([x])

    if not isinstance(x, Iterable):
        raise ValueError(f"{x} is expected to be either a string or an iterable of variable names")

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
