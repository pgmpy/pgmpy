"""
Filtering utilities for tag-based filtering with comparator support.

Used by list_models and list_datasets to support comparator strings for numeric tags:
- ">10", "<20", ">=5", "<=50", "==10"

Invalid comparators (e.g. ">>10", ">abc") and unknown tags return False / empty results;
no exceptions are raised.
"""

import operator
import re
from typing import Any

# Regex to detect comparator strings: >=, <=, ==, >, < followed by a value
_COMPARATOR_PATTERN = re.compile(r"^(>=|<=|==|>|<)(.+)$")

_OPERATOR_MAP = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}


def _is_comparator_condition(condition: Any) -> bool:
    """Return True if condition is a string matching a comparator pattern."""
    if not isinstance(condition, str):
        return False
    return _COMPARATOR_PATTERN.match(condition.strip()) is not None


def _parse_threshold(rhs_str: str) -> int | float | str:
    """Parse the right-hand side of a comparator into int, float, or keep as str."""
    rhs_str = rhs_str.strip()
    try:
        if "." in rhs_str:
            return float(rhs_str)
        return int(rhs_str)
    except ValueError:
        return rhs_str


def apply_filter(value: Any, condition: Any) -> bool:
    """
    Return True if value matches the filter condition, False otherwise.

    Supports both comparator strings (e.g. ">10", "<20", ">=5", "<=50", "==10")
    and exact match conditions.

    Parameters
    ----------
    value : Any
        The metadata value to check (e.g. n_nodes, n_edges).
    condition : Any
        The filter condition. Can be:
        - A comparator string: ">10", "<20", ">=5", "<=50", "==10"
        - Any other value: exact equality (value == condition)

    Returns
    -------
    bool
        True if the value matches the condition.

    Examples
    --------
    >>> apply_filter(15, ">10")
    True
    >>> apply_filter(5, ">10")
    False
    >>> apply_filter(37, 37)
    True
    >>> apply_filter(37, ">=5")
    True
    """
    if condition is None:
        return True

    if isinstance(condition, str):
        match = _COMPARATOR_PATTERN.match(condition.strip())
        if match:
            op_str, rhs_str = match.groups()
            threshold = _parse_threshold(rhs_str)
            op_fn = _OPERATOR_MAP.get(op_str)
            if op_fn is None:
                return value == condition
            if value is None:
                return False
            try:
                return op_fn(value, threshold)
            except TypeError:
                return False

    return value == condition


def split_filter_tags(
    filter_tags: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split filter tags into skbase-compatible (exact match) and custom (comparator) filters.

    Parameters
    ----------
    filter_tags : dict
        Raw filter tags as passed to list_models or list_datasets.

    Returns
    -------
    tuple of (skbase_filters, custom_filters)
        skbase_filters: filters to pass to all_objects (exact match, bool, etc.)
        custom_filters: filters requiring apply_filter (comparator strings)
    """
    skbase_filters = {}
    custom_filters = {}
    for key, value in filter_tags.items():
        if _is_comparator_condition(value):
            custom_filters[key] = value
        else:
            skbase_filters[key] = value
    return skbase_filters, custom_filters
