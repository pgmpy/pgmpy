import random

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
    "causalbench_sachs",
    "causalbench_dream3",
    "causalbench_dream4",
    "lucas0",
    "lucas1",
    "lucas2",
    "lucap0",
    "lucap1",
    "lucap2",
    "cina0",
    "cina1",
    "cina2",
    "reged0",
    "reged1",
    "reged2",
    "sido0",
    "sido1",
    "sido2",
    "marti0",
    "marti1",
    "marti2",
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
    # Filter out causalbench and challenge datasets to avoid downloading large files during random testing
    testable_datasets = [
        d
        for d in ALL_DATASETS
        if not d.startswith("causalbench")
        and d
        not in {
            "lucas0",
            "lucas1",
            "lucas2",
            "lucap0",
            "lucap1",
            "lucap2",
            "cina0",
            "cina1",
            "cina2",
            "reged0",
            "reged1",
            "reged2",
            "sido0",
            "sido1",
            "sido2",
            "marti0",
            "marti1",
            "marti2",
        }
    ]
    for dataset_name in np.random.choice(testable_datasets, size=5, replace=False):
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

        if dataset.tags.get("has_expert_knowledge"):
            assert isinstance(dataset.expert_knowledge, ExpertKnowledge)
        else:
            assert dataset.expert_knowledge is None

        if dataset.tags.get("has_missing_data"):
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


def test_tubingen_missing_data_tag():
    for i in random.sample(range(1, 109), 5):
        dataset = load_dataset(f"tubingen/{i}")
        actual_missing = dataset.data.isnull().any().any()
        assert dataset.tags["has_missing_data"] == actual_missing, (
            f"tubingen/{i}: has_missing_data tag is {dataset.tags['has_missing_data']} "
            f"but actual NaN presence is {actual_missing}"
        )


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


import sys
from unittest.mock import MagicMock, patch


def test_causalbench_datasets_mocked():
    # Use mock (virtual) technique to avoid downloading large datasets and using storage
    mock_causalbench = MagicMock()
    mock_causalbench.load_dataset.return_value = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    with patch.dict(sys.modules, {"causalbench": mock_causalbench}):
        for name in ["causalbench_sachs", "causalbench_dream3", "causalbench_dream4"]:
            dataset = load_dataset(name)
            assert dataset.name == name
            assert isinstance(dataset.data, pd.DataFrame)
            assert dataset.data.shape == (2, 2)
            mock_causalbench.load_dataset.assert_called()


def test_causalbench_datasets_import_error():
    # Force ImportError by mocking sys.modules to None for causalbench
    with patch.dict(sys.modules, {"causalbench": None}):
        for name in ["causalbench_sachs", "causalbench_dream3", "causalbench_dream4"]:
            with pytest.raises(ImportError, match="causalbench is required"):
                load_dataset(name)


def test_causality_challenge_mocked():
    # Use mock technique to avoid downloading large datasets and using storage
    with patch("urllib.request.urlopen") as mock_urlopen:
        # Create a valid dummy zip file in memory containing fake train.data and train.targets
        import io
        import zipfile

        fake_zip = io.BytesIO()
        with zipfile.ZipFile(fake_zip, mode="w") as zf:
            zf.writestr("lucas0_train.data", "1 2 3\n4 5 6\n")
            zf.writestr("lucas0_train.targets", "1\n0\n")

        fake_zip.seek(0)
        mock_response = MagicMock()
        mock_response.read.return_value = fake_zip.read()
        mock_urlopen.return_value = mock_response

        dataset = load_dataset("lucas0")
        assert dataset.name == "lucas0"
        assert dataset.data.shape == (2, 4)  # 3 features + 1 target
        assert "target" in dataset.data.columns
        mock_urlopen.assert_called()
