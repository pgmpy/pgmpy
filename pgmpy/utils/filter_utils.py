"""
Comparator-based filtering utility for pgmpy's list_models() and list_datasets().

Adds Django ORM-style suffix operators on top of skbase's exact-match filter_tags:

    Suffix   Operator
    ------   --------
    __gt     >
    __gte    >=
    __lt
    __lte    <=
    __ne     !=
    __in     value in iterable

Fields with no suffix use the existing exact-match behaviour -- fully
backward-compatible, no existing call breaks.
"""

from __future__ import annotations

import operator
from typing import Any

_SUFFIX_TO_OP: dict[str, Any] = {
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "ne": operator.ne,
}

_ALL_SUFFIXES: frozenset[str] = frozenset(_SUFFIX_TO_OP) | {"in"}


def split_filter_tags(
    kwargs: dict[str, Any],
    valid_tags: set[str],
) -> tuple[dict[str, Any], list[tuple[str, Any, Any]]]:
    """
    Split keyword arguments into exact-match tags and comparator filters.

    Parameters
    ----------
    kwargs : dict
        Raw keyword arguments from the caller.
    valid_tags : set of str
        Legal tag names for this object type.

    Returns
    -------
    exact_tags : dict
        Tags for direct use as ``filter_tags`` in ``all_objects()``.
    comparator_filters : list of (tag_name, op, value)
        op is a binary callable or the string ``"in"``.

    Raises
    ------
    ValueError
        If an unrecognised tag name is used.
    TypeError
        If ``__in`` is given a non-iterable value.
    """
    exact_tags: dict[str, Any] = {}
    comparator_filters: list[tuple[str, Any, Any]] = []

    for key, value in kwargs.items():
        parts = key.rsplit("__", 1)

        if len(parts) == 2 and parts[1] in _ALL_SUFFIXES:
            tag_name, suffix = parts

            if tag_name not in valid_tags:
                raise ValueError(
                    f"Unrecognized filter argument(s): ['{key}']. Valid filter tags are: {sorted(valid_tags)}."
                )

            if suffix == "in":
                if not isinstance(value, (list, tuple, set, frozenset)):
                    raise TypeError(f"Value for '{key}' must be a list, tuple, or set; got {type(value).__name__}.")
                comparator_filters.append((tag_name, "in", value))
            else:
                comparator_filters.append((tag_name, _SUFFIX_TO_OP[suffix], value))

        else:
            if key not in valid_tags:
                raise ValueError(
                    f"Unrecognized filter argument(s): ['{key}']. Valid filter tags are: {sorted(valid_tags)}."
                )
            exact_tags[key] = value

    return exact_tags, comparator_filters


def apply_comparator_filters(
    classes: list[Any],
    comparator_filters: list[tuple[str, Any, Any]],
) -> list[Any]:
    """
    Post-filter a list of skbase class objects using comparator filters.

    Parameters
    ----------
    classes : list
        Objects returned by ``all_objects()``. Each must support
        ``get_class_tag(tag_name)``.
    comparator_filters : list of (tag_name, op, value)
        As produced by :func:`split_filter_tags`.

    Returns
    -------
    list
        Subset of *classes* that satisfy every comparator filter (logical AND).
        Objects whose tag value is ``None`` are excluded when a numeric
        comparator is applied.

    Raises
    ------
    ValueError
        If a comparator cannot be applied for a reason other than a None tag.
    """
    if not comparator_filters:
        return classes

    result = []
    for cls in classes:
        keep = True
        for tag_name, op, value in comparator_filters:
            tag_val = cls.get_class_tag(tag_name)

            # None tag values cannot satisfy any comparator — exclude
            if tag_val is None:
                keep = False
                break

            try:
                if op == "in":
                    if tag_val not in value:
                        keep = False
                        break
                else:
                    if not op(tag_val, value):
                        keep = False
                        break
            except TypeError as exc:
                raise ValueError(
                    f"Cannot apply comparator to tag '{tag_name}': "
                    f"tag value {tag_val!r} is not comparable with {value!r}. "
                    f"Original error: {exc}"
                ) from exc
        if keep:
            result.append(cls)

    return result
