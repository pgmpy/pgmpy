import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.datasets import list_datasets, load_dataset
from pgmpy.estimators import ExpertKnowledge

ALL_DATASETS = [
    "abalone_continuous",
    "abalone_mixed",
    "adult",
    "airfoil",
    "angrist_krueger_qob",
    "algerian_forest",
    "apple_watch_fitbit",
    "auto_mpg",
    "blue_driver",
    "boston_housing",
    "cities",
    "college_plans",
    "contraceptive_method",
    "cover_type",
    "credit_approval",
    "cystic_fibrosis",
    "depression_coping",
    "dropouts",
    "dry_bean",
    "galton_stature",
    "goldberg",
    "hitters",
    "htru2",
    "iq_brain_size",
    "lead",
    "myocardial_infarction",
    "pima_diabetes",
    "pittsburgh_bridges",
    "residential_building",
    "sachs_continuous",
    "sachs_continuous_jittered",
    "sachs_continuous_jittered_logscale",
    "sachs_continuous_logscale",
    "sachs_discrete",
    "sachs_mixed",
    "seoul_bike",
    "south_german_credit",
    "spartina",
    "student_performance",
    "superconductivity",
    "uscrime",
    "wine_quality_red",
    "wine_quality_red_white_mixed",
    "wine_quality_white",
    "yacht_hydrodynamics",
]


def test_list_datasets():
    found_datasets = list_datasets()
    for dataset in ALL_DATASETS:
        assert dataset in found_datasets

    assert "abalone_continuous" not in list_datasets(has_ground_truth=True)

    cont_names = list_datasets(is_continuous=True)

    assert "abalone_continuous" in cont_names
    assert "sachs_discrete" not in cont_names
    assert "abalone_mixed" not in cont_names


def test_load_dataset():
    for dataset_name in np.random.choice(ALL_DATASETS, size=10, replace=False):
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


def test_load_tubingen_dataset():

    for i in [1, 47, 86, 88, 108]:
        dataset = load_dataset(f"tubingen/{i}")

        assert dataset.name == f"tubingen/{i}"
        assert isinstance(dataset.data, pd.DataFrame)
        assert list(dataset.data.columns) == ["x", "y"]

        assert isinstance(dataset.ground_truth, DAG)


def test_tubingen_invalid_format():
    with pytest.raises(ValueError):
        load_dataset("tubingen")
    with pytest.raises(ValueError):
        load_dataset("tubingen/")
    with pytest.raises(ValueError):
        load_dataset("tubingen/abc")
    with pytest.raises(ValueError):
        load_dataset("tubingen/999")


def test_invalid_input():
    with pytest.raises(ValueError):
        load_dataset("non_existent_dataset")


def test_invalid_tag():
    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_datasets(is_paraterized=True)  # typo

    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_datasets(num_samples=100)  # wrong key name entirely
