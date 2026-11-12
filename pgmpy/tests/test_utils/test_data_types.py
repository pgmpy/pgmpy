import numpy as np
import pandas as pd
import pytest

from pgmpy.utils import (
    get_dataset_type,
    preprocess_data,
    split_by_type,
)
from pgmpy.utils.data_types import _column_types


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "float": np.linspace(0.0, 1.0, 6),
            "int8": np.arange(6, dtype="int8"),
            "bool": np.arange(6) % 2 == 0,
            "nullable_bool": pd.array([True, False, None, True, False, True], dtype="boolean"),
            "object": list("ababab"),
            "string": pd.array(list("ababab"), dtype="string"),
            "category": pd.Categorical(list("ababab")),
            "ordered_str": pd.Categorical(list("lhlhlh"), categories=["l", "h"], ordered=True),
            "ordered_int": pd.Categorical([1, 2] * 3, categories=[1, 2], ordered=True),
            "nullable_int": pd.array([1, None, 2, 3, None, 1], dtype="Int64"),
        }
    )


def test_column_types_and_preprocess_data(frame):
    original = frame.copy()
    data, types = preprocess_data(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert _column_types(frame) == {
        "float": "N",
        "int8": "N",
        "bool": "B",
        "nullable_bool": "B",
        "object": "C",
        "string": "C",
        "category": "C",
        "ordered_str": "O",
        "ordered_int": "O",
        "nullable_int": "N",
    }
    assert types == {**_column_types(frame), "bool": "C", "nullable_bool": "C"}
    assert data["ordered_str"].cat.ordered and data["ordered_int"].cat.ordered
    assert split_by_type(list(frame.columns), types) == (
        ["float", "int8", "nullable_int"],
        ["bool", "nullable_bool", "object", "string", "category", "ordered_str", "ordered_int"],
    )
    with pytest.raises(ValueError, match="Couldn't infer datatype of column: when"):
        preprocess_data(pd.DataFrame({"when": pd.date_range("2026-01-01", periods=3)}))


@pytest.mark.parametrize(
    ("columns", "data_types", "dtypes", "dataset_type"),
    [
        (["float", "int8", "nullable_int"], None, ["float64", "int64", "Int64"], "continuous"),
        (["object", "string", "category", "ordered_str", "ordered_int"], None, ["category"] * 5, "discrete"),
        (["float", "object"], None, ["float64", "category"], "mixed"),
        (["bool", "float"], None, ["int64", "float64"], "continuous"),
        (["bool", "object"], None, ["category", "category"], "discrete"),
        (["bool", "nullable_bool"], None, ["category", "category"], "discrete"),
        (["bool", "nullable_bool"], ["continuous"], ["int64", "Int64"], "continuous"),
        (["bool", "nullable_bool"], ["discrete", "continuous"], ["category", "category"], "discrete"),
        (["bool", "object"], ["continuous"], ["category", "category"], "discrete"),
    ],
)
def test_dataset_type_and_bool_resolution(frame, columns, data_types, dtypes, dataset_type):
    data, _ = preprocess_data(frame[columns], data_types=data_types)
    assert data.dtypes.astype(str).tolist() == dtypes
    assert get_dataset_type(data) == dataset_type
    assert get_dataset_type(frame[columns]) == get_dataset_type(preprocess_data(frame[columns])[0])
