"""
Unit tests for pgmpy.utils.filter_utils.

Self-contained -- uses stub classes so no network access or model loading needed.
Tests mirror the style and structure of the existing test_example_models.py and
test_datasets.py in this repo.
"""

import operator

import pytest

from pgmpy.utils.filter_utils import (
    _ALL_SUFFIXES,
    _SUFFIX_TO_OP,
    apply_comparator_filters,
    split_filter_tags,
)

# ---------------------------------------------------------------------------
# Stub skbase-style class
# ---------------------------------------------------------------------------


class _Stub:
    def __init__(self, **tags):
        self._tags = tags

    def get_class_tag(self, name):
        return self._tags.get(name)


def make_stubs(*tag_dicts):
    return [_Stub(**d) for d in tag_dicts]


def names(stubs):
    return {s.get_class_tag("name") for s in stubs}


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_TAGS = {"n_nodes", "n_edges", "is_parameterized", "name"}

STUBS = make_stubs(
    {"name": "A", "n_nodes": 5, "n_edges": 4, "is_parameterized": True},
    {"name": "B", "n_nodes": 10, "n_edges": 9, "is_parameterized": False},
    {"name": "C", "n_nodes": 20, "n_edges": 19, "is_parameterized": True},
    {"name": "D", "n_nodes": 50, "n_edges": 60, "is_parameterized": False},
    {"name": "E", "n_nodes": 10, "n_edges": 10, "is_parameterized": True},
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_suffix_to_op_operators():
    assert _SUFFIX_TO_OP["gt"] is operator.gt
    assert _SUFFIX_TO_OP["gte"] is operator.ge
    assert _SUFFIX_TO_OP["lt"] is operator.lt
    assert _SUFFIX_TO_OP["lte"] is operator.le
    assert _SUFFIX_TO_OP["ne"] is operator.ne


def test_all_suffixes_complete():
    assert "in" in _ALL_SUFFIXES
    for k in _SUFFIX_TO_OP:
        assert k in _ALL_SUFFIXES


# ---------------------------------------------------------------------------
# split_filter_tags -- exact passthrough (backward-compat)
# ---------------------------------------------------------------------------


def test_split_exact_bool():
    exact, comp = split_filter_tags({"is_parameterized": True}, VALID_TAGS)
    assert exact == {"is_parameterized": True} and comp == []


def test_split_exact_int():
    exact, comp = split_filter_tags({"n_nodes": 10}, VALID_TAGS)
    assert exact == {"n_nodes": 10} and comp == []


def test_split_exact_string():
    exact, comp = split_filter_tags({"name": "bnlearn/alarm"}, VALID_TAGS)
    assert exact == {"name": "bnlearn/alarm"} and comp == []


def test_split_multiple_exact():
    exact, comp = split_filter_tags({"n_nodes": 10, "is_parameterized": True}, VALID_TAGS)
    assert exact == {"n_nodes": 10, "is_parameterized": True} and comp == []


def test_split_empty_kwargs():
    exact, comp = split_filter_tags({}, VALID_TAGS)
    assert exact == {} and comp == []


# ---------------------------------------------------------------------------
# split_filter_tags -- comparator suffixes
# ---------------------------------------------------------------------------


def test_split_gt():
    exact, comp = split_filter_tags({"n_nodes__gt": 10}, VALID_TAGS)
    assert exact == {}
    tag, op, val = comp[0]
    assert tag == "n_nodes" and op is operator.gt and val == 10


def test_split_gte():
    _, comp = split_filter_tags({"n_nodes__gte": 10}, VALID_TAGS)
    assert comp[0][1] is operator.ge


def test_split_lt():
    _, comp = split_filter_tags({"n_nodes__lt": 10}, VALID_TAGS)
    assert comp[0][1] is operator.lt


def test_split_lte():
    _, comp = split_filter_tags({"n_nodes__lte": 10}, VALID_TAGS)
    assert comp[0][1] is operator.le


def test_split_ne():
    _, comp = split_filter_tags({"n_nodes__ne": 10}, VALID_TAGS)
    assert comp[0][1] is operator.ne


def test_split_in_list():
    exact, comp = split_filter_tags({"n_nodes__in": [10, 20]}, VALID_TAGS)
    assert exact == {}
    tag, op, val = comp[0]
    assert tag == "n_nodes" and op == "in" and val == [10, 20]


def test_split_in_tuple():
    _, comp = split_filter_tags({"n_nodes__in": (10, 20)}, VALID_TAGS)
    assert comp[0][1] == "in"


def test_split_in_set():
    _, comp = split_filter_tags({"n_nodes__in": {10, 20}}, VALID_TAGS)
    assert comp[0][1] == "in"


def test_split_two_comparators_same_field():
    exact, comp = split_filter_tags({"n_nodes__gte": 10, "n_nodes__lte": 50}, VALID_TAGS)
    assert exact == {} and len(comp) == 2


def test_split_mixed_exact_and_comparator():
    exact, comp = split_filter_tags({"is_parameterized": True, "n_nodes__gt": 5}, VALID_TAGS)
    assert exact == {"is_parameterized": True} and len(comp) == 1


# ---------------------------------------------------------------------------
# split_filter_tags -- error cases (mirrors test_invalid_tag in repo)
# ---------------------------------------------------------------------------


def test_split_unknown_exact_tag_raises():
    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        split_filter_tags({"num_nodes": 10}, VALID_TAGS)


def test_split_unknown_base_tag_with_suffix_raises():
    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        split_filter_tags({"num_nodes__gt": 10}, VALID_TAGS)


def test_split_typo_raises():
    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        split_filter_tags({"is_paraterized": True}, VALID_TAGS)


def test_split_in_scalar_raises():
    with pytest.raises(TypeError, match="list, tuple, or set"):
        split_filter_tags({"n_nodes__in": 10}, VALID_TAGS)


def test_split_in_string_raises():
    with pytest.raises(TypeError, match="list, tuple, or set"):
        split_filter_tags({"n_nodes__in": "10"}, VALID_TAGS)


# ---------------------------------------------------------------------------
# apply_comparator_filters
# ---------------------------------------------------------------------------


def test_apply_no_filters_returns_all():
    assert apply_comparator_filters(STUBS, []) == STUBS


def test_apply_empty_input():
    assert apply_comparator_filters([], [("n_nodes", operator.gt, 5)]) == []


def test_apply_gt():
    result = apply_comparator_filters(STUBS, [("n_nodes", operator.gt, 10)])
    assert names(result) == {"C", "D"}


def test_apply_gte():
    result = apply_comparator_filters(STUBS, [("n_nodes", operator.ge, 10)])
    assert names(result) == {"B", "C", "D", "E"}


def test_apply_lt():
    result = apply_comparator_filters(STUBS, [("n_nodes", operator.lt, 10)])
    assert names(result) == {"A"}


def test_apply_lte():
    result = apply_comparator_filters(STUBS, [("n_nodes", operator.le, 10)])
    assert names(result) == {"A", "B", "E"}


def test_apply_ne():
    result = apply_comparator_filters(STUBS, [("n_nodes", operator.ne, 10)])
    assert names(result) == {"A", "C", "D"}


def test_apply_in_list():
    result = apply_comparator_filters(STUBS, [("n_nodes", "in", [5, 50])])
    assert names(result) == {"A", "D"}


def test_apply_in_tuple():
    result = apply_comparator_filters(STUBS, [("n_nodes", "in", (5, 50))])
    assert names(result) == {"A", "D"}


def test_apply_in_set():
    result = apply_comparator_filters(STUBS, [("n_nodes", "in", {5, 50})])
    assert names(result) == {"A", "D"}


def test_apply_range_gte_lte():
    filters = [("n_nodes", operator.ge, 10), ("n_nodes", operator.le, 20)]
    assert names(apply_comparator_filters(STUBS, filters)) == {"B", "C", "E"}


def test_apply_range_gt_lt():
    filters = [("n_nodes", operator.gt, 5), ("n_nodes", operator.lt, 20)]
    assert names(apply_comparator_filters(STUBS, filters)) == {"B", "E"}


def test_apply_range_no_results():
    filters = [("n_nodes", operator.gt, 50), ("n_nodes", operator.lt, 100)]
    assert apply_comparator_filters(STUBS, filters) == []


def test_apply_numeric_and_bool():
    filters = [
        ("n_nodes", operator.gt, 5),
        ("is_parameterized", operator.eq, True),
    ]
    assert names(apply_comparator_filters(STUBS, filters)) == {"C", "E"}


def test_apply_no_match_returns_empty():
    result = apply_comparator_filters(STUBS, [("n_nodes", operator.gt, 9999)])
    assert result == []


def test_apply_none_tag_value_excluded():
    stubs = make_stubs({"name": "X", "n_nodes": None})
    result = apply_comparator_filters(stubs, [("n_nodes", operator.gt, 5)])
    assert result == []


def test_apply_bad_type_raises():
    stubs = make_stubs({"name": "X", "n_nodes": "not_a_number"})
    with pytest.raises(ValueError, match="Cannot apply comparator"):
        apply_comparator_filters(stubs, [("n_nodes", operator.gt, 5)])


# ---------------------------------------------------------------------------
# End-to-end pipeline: split_filter_tags -> apply_comparator_filters
# (mirrors what list_models / list_datasets do internally)
# ---------------------------------------------------------------------------


def _run(**kwargs):
    """Simulate list_models: split -> exact filter -> comparator filter."""
    exact, comp = split_filter_tags(kwargs, VALID_TAGS)
    after_exact = [s for s in STUBS if all(s.get_class_tag(k) == v for k, v in exact.items())]
    return apply_comparator_filters(after_exact, comp)


def test_pipeline_no_kwargs_returns_all():
    assert len(_run()) == len(STUBS)


def test_pipeline_exact_int():
    assert names(_run(n_nodes=10)) == {"B", "E"}


def test_pipeline_exact_bool():
    assert names(_run(is_parameterized=True)) == {"A", "C", "E"}


def test_pipeline_exact_int_and_bool():
    assert names(_run(n_nodes=10, is_parameterized=True)) == {"E"}


def test_pipeline_gt():
    assert names(_run(n_nodes__gt=10)) == {"C", "D"}


def test_pipeline_gte():
    assert names(_run(n_nodes__gte=10)) == {"B", "C", "D", "E"}


def test_pipeline_lt():
    assert names(_run(n_nodes__lt=10)) == {"A"}


def test_pipeline_lte():
    assert names(_run(n_nodes__lte=10)) == {"A", "B", "E"}


def test_pipeline_ne():
    assert names(_run(n_nodes__ne=10)) == {"A", "C", "D"}


def test_pipeline_in():
    assert names(_run(n_nodes__in=[5, 50])) == {"A", "D"}


def test_pipeline_range_gte_lte():
    assert names(_run(n_nodes__gte=10, n_nodes__lte=20)) == {"B", "C", "E"}


def test_pipeline_range_gt_lt():
    assert names(_run(n_nodes__gt=5, n_nodes__lt=20)) == {"B", "E"}


def test_pipeline_mixed_exact_and_comparator():
    assert names(_run(is_parameterized=True, n_nodes__gt=5)) == {"C", "E"}


def test_pipeline_exact_bool_and_range():
    assert names(_run(is_parameterized=True, n_nodes__gte=10, n_nodes__lte=20)) == {
        "C",
        "E",
    }


def test_pipeline_no_match():
    assert _run(n_nodes__gt=9999) == []


def test_pipeline_in_with_bool():
    assert names(_run(n_nodes__in=[5, 10], is_parameterized=True)) == {"A", "E"}


def test_pipeline_invalid_tag_raises():
    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        split_filter_tags({"nonexistent": 1}, VALID_TAGS)


def test_pipeline_invalid_tag_with_suffix_raises():
    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        split_filter_tags({"nonexistent__gt": 1}, VALID_TAGS)


def test_pipeline_in_scalar_raises():
    with pytest.raises(TypeError, match="list, tuple, or set"):
        split_filter_tags({"n_nodes__in": 42}, VALID_TAGS)
