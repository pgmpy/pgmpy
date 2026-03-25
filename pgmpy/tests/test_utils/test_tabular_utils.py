import pandas as pd
import pytest

from pgmpy.utils import build_state_names, collect_state_names, get_state_counts


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


def test_get_state_counts_missing_weight_column_error(dataset):
    state_names = build_state_names(dataset)

    with pytest.raises(ValueError, match="data must contain a `_weight` column if weighted=True"):
        get_state_counts(dataset, state_names, variable="A", weighted=True)
