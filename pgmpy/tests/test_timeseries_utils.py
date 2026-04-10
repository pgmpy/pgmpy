import numpy as np
import pandas as pd
import pytest

from pgmpy.utils import from_dbn_to_sktime, from_sktime_to_dbn


@pytest.fixture(scope="module")
def sample_panel_df():
    """Create a simple 2-instance, 3-time-step panel dataframe."""
    idx = pd.MultiIndex.from_product(
        [[0, 1], ["t0", "t1", "t2"]], names=["instance", "time"]
    )
    data = {
        "A": np.arange(6),
        "B": np.arange(6, 12),
    }
    return pd.DataFrame(data, index=idx)


def test_multiindex_roundtrip(sample_panel_df):
    dbn_df = from_sktime_to_dbn(sample_panel_df)

    # Expect 2 instances (rows)
    assert dbn_df.shape[0] == 2
    # Expect 2 vars * 3 time slices = 6 columns, each a tuple
    assert dbn_df.shape[1] == 6
    assert all(isinstance(c, tuple) and len(c) == 2 for c in dbn_df.columns)

    # Convert back
    recovered = from_dbn_to_sktime(dbn_df)
    # Sort to align order with original
    recovered = recovered.sort_index()
    original = sample_panel_df.sort_index()
    pd.testing.assert_frame_equal(recovered, original)


def test_instance_col_roundtrip():
    # Build wide df with DateTime index and instance_col column
    time_index = pd.date_range("2021-01-01", periods=3, freq="D")
    df_list = []
    for inst in ["x", "y"]:
        tmp = pd.DataFrame(
            {
                "instance": inst,
                "A": np.random.randn(3),
                "B": np.random.randn(3),
            },
            index=time_index,
        )
        df_list.append(tmp)
    wide_df = pd.concat(df_list)

    dbn_df = from_sktime_to_dbn(wide_df, instance_col="instance")
    recovered = from_dbn_to_sktime(dbn_df)

    # Align ordering for comparison
    recovered = recovered.sort_index()
    wide_df_sorted = (
        wide_df.set_index(["instance", wide_df.index])
        .sort_index()
        .rename_axis(["instance", "time"])
    )
    pd.testing.assert_frame_equal(recovered, wide_df_sorted)


# ---------------------------------------------------------------------------
# Edge-case tests for uncovered branches
# ---------------------------------------------------------------------------


class TestFromSktimeToDbnEdgeCases:
    """Tests for error-handling and edge-case branches in from_sktime_to_dbn."""

    def test_multiindex_non_2_level_raises(self):
        """3-level MultiIndex should raise ValueError (lines 92-95)."""
        idx = pd.MultiIndex.from_tuples(
            [(0, "a", 1), (0, "b", 2)], names=["x", "y", "z"]
        )
        df = pd.DataFrame({"A": [1, 2]}, index=idx)
        with pytest.raises(ValueError, match="exactly 2 levels"):
            from_sktime_to_dbn(df)

    def test_single_trajectory_no_instance_col(self):
        """Plain DataFrame without instance_col => single trajectory (lines 99-105)."""
        df = pd.DataFrame(
            {"A": [10, 20, 30], "B": [1.0, 2.0, 3.0]},
            index=pd.RangeIndex(3),
        )
        result = from_sktime_to_dbn(df)
        assert result.shape == (1, 6)  # 1 instance, 2 vars * 3 time steps
        assert all(isinstance(c, tuple) for c in result.columns)
        assert result.iloc[0][("A", 0)] == 10
        assert result.iloc[0][("B", 2)] == 3.0

    def test_single_trajectory_roundtrip(self):
        """Single trajectory should survive round-trip (lines 99-105)."""
        df = pd.DataFrame(
            {"A": [10, 20], "B": [1.0, 2.0]},
            index=[100, 200],
        )
        dbn_df = from_sktime_to_dbn(df)
        recovered = from_dbn_to_sktime(dbn_df)
        assert recovered.shape == (2, 2)
        assert recovered.index.names == ["instance", "time"]

    def test_missing_instance_col_raises(self):
        """Non-existent instance_col should raise KeyError (line 108)."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with pytest.raises(KeyError, match="instance_col.*not_a_column"):
            from_sktime_to_dbn(df, instance_col="not_a_column")

    def test_missing_time_steps_filled_with_nan(self):
        """Unbalanced panels should fill missing time steps with NaN (lines 139-142)."""
        idx = pd.MultiIndex.from_tuples(
            [(0, "t0"), (0, "t1"), (0, "t2"), (1, "t0"), (1, "t2")],
            names=["instance", "time"],
        )
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4, 6], "B": [10, 20, 30, 40, 60]}, index=idx
        )
        result = from_sktime_to_dbn(df)
        assert result.shape == (2, 6)  # 2 vars * 3 time steps
        # Access tuple columns via [] then row with .loc
        assert np.isnan(result[("A", 1)].loc[1])
        assert np.isnan(result[("B", 1)].loc[1])
        assert result[("A", 1)].loc[0] == 2


class TestFromDbnToSktimeEdgeCases:
    """Tests for error-handling branches in from_dbn_to_sktime."""

    def test_non_tuple_columns_raises(self):
        """Plain string columns should raise ValueError (line 178)."""
        df = pd.DataFrame({"A": [1], "B": [2]})
        with pytest.raises(ValueError, match="2-tuples"):
            from_dbn_to_sktime(df)

    def test_tuple_with_wrong_length_raises(self):
        """3-tuples should raise ValueError (line 178)."""
        dbn_df = pd.DataFrame(
            {("A", 0, "extra"): [1], ("B", 1, "extra"): [2]}
        )
        with pytest.raises(ValueError, match="2-tuples"):
            from_dbn_to_sktime(dbn_df)

    def test_without_time_reverse_map(self):
        """Missing _time_reverse_map should still work, using int time (lines 192-197)."""
        dbn_df = pd.DataFrame(
            {
                ("A", 0): [1, 4],
                ("A", 1): [2, 5],
                ("B", 0): [3, 6],
                ("B", 1): [7, 8],
            }
        )
        assert "_time_reverse_map" not in dbn_df.attrs
        result = from_dbn_to_sktime(dbn_df)
        assert result.index.names == ["instance", "time"]
        assert set(result.index.get_level_values("time")) == {0, 1}
