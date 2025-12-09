import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import DAG
from pgmpy.datasets import DATASETS, load_dataset


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_registry():
    """Test if datasets are registered correctly."""
    datasets = DATASETS.list_all()
    assert "abalone" in datasets
    assert "sachs" in datasets


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_load_dataset_return_types():
    """Test the conditional return types of load_dataset."""
    df, ground_truth = load_dataset("sachs")
    assert isinstance(df, pd.DataFrame)
    assert isinstance(ground_truth, DAG)

    # 2. load_ground_truth=False -> returns DataFrame only
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
    assert len(ground_truth.nodes()) == 20


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_invalid_input():
    with pytest.raises(ValueError):
        load_dataset("non_existent_dataset")

    with pytest.raises(KeyError):
        load_dataset("sachs", variant="bad_variant")
