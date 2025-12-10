import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import DAG
from pgmpy.datasets import DATASET_REGISTRY, SachsContinuous, load_dataset
from pgmpy.estimators import ExpertKnowledge

ALL_DATASETS = [
    "abalone_continuous",
    "sachs_continuous",
    "sachs_discrete",
    "sachs_continuous_logscale",
]


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_list_datasets():
    datasets = DATASET_REGISTRY.list_datasets()
    for dataset in ALL_DATASETS:
        assert dataset in datasets

    datasets_filtered = DATASET_REGISTRY.list_datasets(has_ground_truth=True)
    for dataset in ["sachs_continuous", "sachs_discrete"]:
        assert dataset in datasets_filtered
    for dataset in ["abalone_continuous", "abalone_mixed"]:
        assert dataset not in datasets_filtered

    datasets_filtered = DATASET_REGISTRY.list_datasets(is_continuous=True)
    assert "sachs_continuous" in datasets_filtered
    assert "sachs_discrete" not in datasets_filtered


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_load_dataset():
    dataset = load_dataset("sachs_continuous")
    assert dataset.name == "sachs_continuous"
    assert isinstance(dataset.data, pd.DataFrame)
    assert isinstance(dataset.ground_truth, DAG)
    assert dataset.tags == SachsContinuous.tags
    assert isinstance(dataset.expert_knowledge, ExpertKnowledge)


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_invalid_input():
    with pytest.raises(ValueError):
        load_dataset("non_existent_dataset")
