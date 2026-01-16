import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import DAG
from pgmpy.datasets import list_datasets, load_dataset
from pgmpy.estimators import ExpertKnowledge

ALL_DATASETS = [
    "abalone_continuous",
    "abalone_mixed",
    "adult",
    "airfoil",
    "algerian_forest",
    "apple_watch_fitbit",
    "auto_mpg",
    "boston_housing",
    "cities",
    "college_plans",
    "lead",
    "spartina",
    "goldberg",
    "hitters",
    "pittsburgh_bridges",
    "residential_building",
    "credit_approval",
    "depression_coping",
    "iq_brain_size",
    "contraceptive_method",
    "myocardial_infarction",
    "htru2",
    "dry_bean",
    "cystic_fibrosis",
    "south_german_credit",
    "pima_diabetes",
    "galton_stature",
    "sachs_mixed",
    "sachs_continuous",
    "sachs_discrete",
    "sachs_continuous_logscale",
    "sachs_continuous_jittered_logscale",
    "sachs_continuous_jittered",
    "superconductivity",
    "yacht_hydrodynamics",
    "student_performance",
    "seoul_bike",
    "wine_quality_red",
    "wine_quality_white",
    "wine_quality_red_white_mixed",
]


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_list_datasets():
    found_datasets = list_datasets()
    for dataset in ALL_DATASETS:
        assert dataset in found_datasets

    assert "abalone_continuous" not in list_datasets(has_ground_truth=True)

    cont_names = list_datasets(is_continuous=True)

    assert "abalone_continuous" in cont_names
    assert "sachs_discrete" not in cont_names
    assert "abalone_mixed" not in cont_names


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_load_dataset():
    for dataset_name in np.random.choice(ALL_DATASETS, size=5, replace=False):
        dataset = load_dataset(dataset_name)
        assert dataset.name == dataset_name
        assert dataset.data.shape == (
            dataset.tags["n_samples"],
            dataset.tags["n_variables"],
        )
        assert isinstance(dataset.data, pd.DataFrame)
        assert isinstance(dataset.tags, dict)

        if dataset.tags["has_ground_truth"]:
            assert isinstance(dataset.ground_truth, DAG)
        else:
            assert dataset.ground_truth is None

        if dataset.tags["has_expert_knowledge"]:
            assert isinstance(dataset.expert_knowledge, ExpertKnowledge)
        else:
            assert dataset.expert_knowledge is None

        if dataset.tags["has_missing_data"]:
            assert dataset.data.isna().any().any()


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_load_covariance_dataset():
    for name in ["goldberg", "spartina", "lead", "cities"]:
        dataset = load_dataset(name)
        assert dataset.name == name
        assert dataset.data.shape == (
            dataset.tags["n_samples"],
            dataset.tags["n_variables"],
        )
        assert isinstance(dataset.data, pd.DataFrame)
        assert isinstance(dataset.tags, dict)


@pytest.mark.skipif(
    not _check_soft_dependencies("requests", severity="none"),
    reason="test only if requests is installed",
)
def test_invalid_input():
    with pytest.raises(ValueError):
        load_dataset("non_existent_dataset")
