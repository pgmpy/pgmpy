import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import DAG
from pgmpy.datasets import DATASETS, load_dataset

ALL_DATASETS = ["abalone", "sachs"]


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_list_datasets():
    datasets = DATASETS.list_datasets()
    for dataset in ALL_DATASETS:
        assert dataset in datasets

    datasets_filtered = DATASETS.list_datasets(has_ground_truth=True)
    for dataset in ["abalone", "sachs"]:
        assert dataset in datasets_filtered

    datasets_filtered = DATASETS.list_datasets(is_continuous=True)
    assert "sachs" in datasets_filtered


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_load_dataset():
    df, ground_truth = load_dataset("sachs")
    assert isinstance(df, pd.DataFrame)
    assert isinstance(ground_truth, DAG)

    df_only = load_dataset("sachs", load_ground_truth=False)
    assert isinstance(df_only, pd.DataFrame)
    assert not isinstance(df_only, tuple)


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_sachs_jittered_variant():
    """Test specific variant logic where nodes mismatch."""
    df, ground_truth = load_dataset("sachs", variant="jittered_experimental")

    assert df.shape[1] == 20


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_invalid_input():
    with pytest.raises(ValueError):
        load_dataset("non_existent_dataset")

    with pytest.raises(ValueError):
        load_dataset("sachs", variant="bad_variant")
