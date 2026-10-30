import numpy as np
import pandas as pd
import pytest

from pgmpy.utils import (
    build_state_names,
    collect_state_names,
    encode_columns,
    get_state_counts,
    get_state_counts_array,
)


@pytest.fixture
def dataset():
    return pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]})


def test_collect_state_names(dataset):
    assert collect_state_names(dataset, "A") == [0, 1]
    assert collect_state_names(dataset, "D") == ["X", "Y", "Z"]


def test_build_state_names(dataset):
    state_names = build_state_names(dataset)

    assert state_names == {
        "A": [0, 1],
        "B": [0, 1],
        "C": [0, 1],
        "D": ["X", "Y", "Z"],
    }


def test_build_state_names_with_custom_states(dataset):
    state_names = build_state_names(dataset, state_names={"A": [0, 1, 2]})

    assert state_names["A"] == [0, 1, 2]
    assert state_names["B"] == [0, 1]


def test_build_state_names_raises_on_unexpected_state(dataset):
    with pytest.raises(ValueError, match="Data contains unexpected states for variable: A\\."):
        build_state_names(dataset, state_names={"A": [0]})


def test_get_state_counts(dataset):
    state_names = build_state_names(dataset)

    assert get_state_counts(dataset, state_names, variable="A").values.tolist() == [[2], [1]]
    assert get_state_counts(dataset, state_names, variable="C", parents=("A", "B")).values.tolist() == [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]


def test_get_state_counts_with_sample_weight(dataset):
    state_names = build_state_names(dataset)
    sample_weight = np.array([2.0, 1.0, 3.0])

    assert get_state_counts(dataset, state_names, variable="A", sample_weight=sample_weight).values.tolist() == [
        [3.0],
        [3.0],
    ]
    assert get_state_counts(
        dataset, state_names, variable="C", parents=("A", "B"), sample_weight=sample_weight
    ).values.tolist() == [
        [0.0, 0.0, 3.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
    ]


def test_encode_columns(dataset):
    state_names = build_state_names(dataset)
    codes, cards = encode_columns(dataset, state_names)

    assert cards == {"A": 2, "B": 2, "C": 2, "D": 3}
    assert codes["A"].tolist() == [0, 0, 1]
    assert codes["B"].tolist() == [0, 1, 0]
    assert codes["C"].tolist() == [1, 1, 0]
    assert codes["D"].tolist() == [0, 1, 2]


def test_encode_columns_marks_unknown_as_negative_one():
    df = pd.DataFrame({"A": [1, 2, np.nan, 1]})
    state_names = {"A": [1, 2]}
    codes, cards = encode_columns(df, state_names)
    assert codes["A"].tolist() == [0, 1, -1, 0]
    assert cards["A"] == 2


def test_get_state_counts_array_matches_get_state_counts(dataset):
    """Fast bincount path must produce the same counts as the pandas groupby path."""
    state_names = build_state_names(dataset)
    codes, cards = encode_columns(dataset, state_names)

    # No parents.
    expected_a = get_state_counts(dataset, state_names, variable="A").values
    actual_a = get_state_counts_array(codes, cards, variable="A")
    np.testing.assert_array_equal(actual_a, expected_a)

    # With parents (reindex=True is the canonical full-product layout).
    expected_c = get_state_counts(dataset, state_names, variable="C", parents=("A", "B")).values
    actual_c = get_state_counts_array(codes, cards, variable="C", parents=("A", "B"))
    np.testing.assert_array_equal(actual_c, expected_c)


def test_get_state_counts_array_with_sample_weight(dataset):
    state_names = build_state_names(dataset)
    codes, cards = encode_columns(dataset, state_names)
    sample_weight = np.array([2.0, 1.0, 3.0])

    expected = get_state_counts(
        dataset, state_names, variable="C", parents=("A", "B"), sample_weight=sample_weight
    ).values
    actual = get_state_counts_array(codes, cards, variable="C", parents=("A", "B"), sample_weight=sample_weight)
    np.testing.assert_array_equal(actual, expected)


def test_get_state_counts_array_skips_nan_rows():
    df = pd.DataFrame({"A": [0, 0, 1, np.nan, 1], "B": [0, 1, np.nan, 1, 0]})
    state_names = {"A": [0, 1], "B": [0, 1]}
    codes, cards = encode_columns(df, state_names)

    # Reference: pandas-based count drops the NaN-containing rows by default.
    expected = get_state_counts(df, state_names, variable="A", parents=("B",)).values
    actual = get_state_counts_array(codes, cards, variable="A", parents=("B",))
    np.testing.assert_array_equal(actual, expected)
