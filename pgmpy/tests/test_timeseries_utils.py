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
