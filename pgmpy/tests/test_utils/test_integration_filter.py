"""
Integration tests for comparator-based filtering in list_models and list_datasets.

These tests run against the real model and dataset registry to verify
correctness end-to-end. No network access needed for list_* calls.
"""

import pytest

from pgmpy.datasets import list_datasets
from pgmpy.example_models import list_models

# ---------------------------------------------------------------------------
# list_models — backward compatibility
# ---------------------------------------------------------------------------


def test_list_models_returns_all():
    all_models = list_models()
    assert len(all_models) == 257
    assert all(isinstance(m, str) for m in all_models)
    assert all_models == sorted(all_models)


def test_list_models_exact_bool():
    assert "bnlearn/alarm" in list_models(is_parameterized=True)
    assert "bnlearn/alarm" in list_models(is_discrete=True)
    assert "bnlearn/arth150" in list_models(is_continuous=True)
    assert "bnlearn/arth150" not in list_models(is_discrete=True)


def test_list_models_exact_name():
    assert list_models(name="bnlearn/alarm") == ["bnlearn/alarm"]


# ---------------------------------------------------------------------------
# list_models — comparator suffixes
# ---------------------------------------------------------------------------


def test_list_models_gt():
    r = list_models(n_nodes__gt=10)
    assert len(r) > 0
    assert all(isinstance(m, str) for m in r)


def test_list_models_gte_larger_than_gt():
    assert len(list_models(n_nodes__gte=10)) >= len(list_models(n_nodes__gt=10))


def test_list_models_lt():
    assert len(list_models(n_nodes__lt=10)) > 0


def test_list_models_lte_larger_than_lt():
    assert len(list_models(n_nodes__lte=10)) >= len(list_models(n_nodes__lt=10))


def test_list_models_ne():
    eq = set(list_models(n_nodes=10))
    ne = set(list_models(n_nodes__ne=10))
    all_m = set(list_models())
    assert eq.isdisjoint(ne)
    assert eq | ne == all_m


def test_list_models_in():
    r = list_models(n_nodes__in=[10, 20, 46])
    assert isinstance(r, list)
    assert len(r) > 0


def test_list_models_in_empty():
    assert list_models(n_nodes__in=[]) == []


def test_list_models_impossible_range():
    assert list_models(n_nodes__gt=9999) == []
    assert list_models(n_nodes__gt=50, n_nodes__lt=10) == []


# ---------------------------------------------------------------------------
# list_models — partition correctness
# ---------------------------------------------------------------------------


def test_list_models_gt_lte_partition():
    all_m = set(list_models())
    gt10 = set(list_models(n_nodes__gt=10))
    lte10 = set(list_models(n_nodes__lte=10))
    assert gt10.isdisjoint(lte10)
    assert gt10 | lte10 == all_m


def test_list_models_gte_lt_partition():
    all_m = set(list_models())
    gte10 = set(list_models(n_nodes__gte=10))
    lt10 = set(list_models(n_nodes__lt=10))
    assert gte10.isdisjoint(lt10)
    assert gte10 | lt10 == all_m


# ---------------------------------------------------------------------------
# list_models — range queries
# ---------------------------------------------------------------------------


def test_list_models_range():
    gte10 = set(list_models(n_nodes__gte=10))
    r = list_models(n_nodes__gte=10, n_nodes__lte=50)
    assert len(r) > 0
    assert set(r).issubset(gte10)


# ---------------------------------------------------------------------------
# list_models — mixed exact + comparator
# ---------------------------------------------------------------------------


def test_list_models_mixed():
    gt10 = set(list_models(n_nodes__gt=10))
    discrete = set(list_models(is_discrete=True))
    r = set(list_models(is_discrete=True, n_nodes__gt=10))
    assert r.issubset(discrete)
    assert r.issubset(gt10)


# ---------------------------------------------------------------------------
# list_models — error handling
# ---------------------------------------------------------------------------


def test_list_models_invalid_tag_raises():
    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_models(is_paraterized=True)

    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_models(num_nodes=10)

    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_models(num_nodes__gt=10)


def test_list_models_in_scalar_raises():
    with pytest.raises(TypeError, match="list, tuple, or set"):
        list_models(n_nodes__in=10)

    with pytest.raises(TypeError, match="list, tuple, or set"):
        list_models(n_nodes__in="abc")


# ---------------------------------------------------------------------------
# list_datasets — backward compatibility
# ---------------------------------------------------------------------------


def test_list_datasets_returns_all():
    all_ds = list_datasets()
    assert len(all_ds) > 0
    assert all(isinstance(d, str) for d in all_ds)
    assert all_ds == sorted(all_ds)


def test_list_datasets_exact_bool():
    assert "abalone_continuous" in list_datasets(is_continuous=True)
    assert "sachs_discrete" not in list_datasets(is_continuous=True)
    assert "abalone_mixed" not in list_datasets(is_continuous=True)
    assert "abalone_continuous" not in list_datasets(has_ground_truth=True)
    assert "sachs_discrete" in list_datasets(is_discrete=True, has_ground_truth=True)


# ---------------------------------------------------------------------------
# list_datasets — comparator suffixes
# ---------------------------------------------------------------------------


def test_list_datasets_gt():
    r = list_datasets(n_samples__gt=1000)
    assert len(r) > 0
    assert all(isinstance(d, str) for d in r)


def test_list_datasets_gte_larger_than_gt():
    assert len(list_datasets(n_samples__gte=1000)) >= len(list_datasets(n_samples__gt=1000))


def test_list_datasets_lt():
    assert len(list_datasets(n_samples__lt=1000)) > 0


def test_list_datasets_variables_range():
    r = list_datasets(n_variables__gte=5, n_variables__lte=20)
    assert len(r) > 0


def test_list_datasets_in():
    r = list_datasets(n_variables__in=[5, 10, 15])
    assert isinstance(r, list)


def test_list_datasets_impossible_range():
    assert list_datasets(n_samples__gt=999999) == []


# ---------------------------------------------------------------------------
# list_datasets — mixed exact + comparator
# ---------------------------------------------------------------------------


def test_list_datasets_mixed():
    continuous = set(list_datasets(is_continuous=True))
    r = set(list_datasets(is_continuous=True, n_samples__gt=500))
    assert r.issubset(continuous)


# ---------------------------------------------------------------------------
# list_datasets — error handling
# ---------------------------------------------------------------------------


def test_list_datasets_invalid_tag_raises():
    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_datasets(is_paraterized=True)

    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_datasets(num_samples=100)

    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_datasets(num_samples__gt=100)


def test_list_datasets_in_scalar_raises():
    with pytest.raises(TypeError, match="list, tuple, or set"):
        list_datasets(n_samples__in=100)
