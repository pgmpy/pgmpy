import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import DAG
from pgmpy.datasets import DATASET_REGISTRY, load_dataset
from pgmpy.estimators import ExpertKnowledge

ALL_DATASETS = [
    "abalone_continuous",
    "abalone_mixed",
    "adult",
    "airfoil",
    "algeria_forest",
    "boston_housing",
    "galton_stature",
    "sachs_mixed",
    "sachs_continuous",
    "sachs_discrete",
    "sachs_continuous_logscale",
    "sachs_continuous_jittered_logscale",
    "sachs_continuous_jittered",
    "wine_quality_red",
    "wine_quality_white",
    "wine_quality_red_white_mixed",
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
    assert "abalone_continuous" in datasets_filtered
    assert "sachs_discrete" not in datasets_filtered
    assert "abalone_mixed" not in datasets_filtered


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_load_dataset():
    for dataset_name in np.random.choice(ALL_DATASETS, size=5, replace=False):
        dataset = load_dataset(dataset_name)
        assert dataset.name == dataset_name
        assert isinstance(dataset.data, pd.DataFrame)
        assert isinstance(dataset.tags, dict)

        if DATASET_REGISTRY.get_dataset(dataset_name).tags["has_ground_truth"]:
            assert isinstance(dataset.ground_truth, DAG)
        else:
            assert dataset.ground_truth is None

        if DATASET_REGISTRY.get_dataset(dataset_name).tags["has_expert_knowledge"]:
            assert isinstance(dataset.expert_knowledge, ExpertKnowledge)
        else:
            assert dataset.expert_knowledge is None


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_invalid_input():
    with pytest.raises(ValueError):
        load_dataset("non_existent_dataset")
