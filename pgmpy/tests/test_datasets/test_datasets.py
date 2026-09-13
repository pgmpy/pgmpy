import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.datasets import list_datasets, load_dataset

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
    "car_evaluation",
    "cities",
    "college_plans",
    "contraceptive_method",
    "cover_type",
    "credit_approval",
    "cystic_fibrosis",
    "depression_coping",
    "dropouts",
    "dry_bean",
    "feedbacks_network1_amp",              
    "feedbacks_network2_amp",
    "feedbacks_network3_amp",
    "feedbacks_network4_amp",
    "feedbacks_network5_amp",
    "feedbacks_network5_cont",
    "feedbacks_network5_cont_p3n7",
    "feedbacks_network5_cont_p7n3",
    "feedbacks_network6_amp",
    "feedbacks_network6_cont",
    "feedbacks_network7_amp",
    "feedbacks_network7_cont",
    "feedbacks_network8_amp_amp",
    "feedbacks_network8_amp_cont",
    "feedbacks_network8_cont_amp",
    "feedbacks_network9_amp_amp",
    "feedbacks_network9_amp_cont",
    "feedbacks_network9_cont_amp",
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
    for dataset_name in ["sachs_discrete"]:
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

    # n_samples controls generated covariance dataset size.
    dataset = load_dataset("goldberg", n_samples=25, seed=42)
    assert dataset.data.shape[0] == 25

    # The same seed produces identical generated data.
    ds1 = load_dataset("goldberg", seed=42)
    ds2 = load_dataset("goldberg", seed=42)
    pd.testing.assert_frame_equal(ds1.data, ds2.data)


def test_load_tubingen_dataset():
    for i in [1, 47, 108]:
        dataset = load_dataset(f"tubingen/{i}")

        assert dataset.name == f"tubingen/{i}"
        assert isinstance(dataset.data, pd.DataFrame)
        assert list(dataset.data.columns) == ["x", "y"]

        assert isinstance(dataset.ground_truth, DAG)

    # Tubingen supports n_samples by subsampling the selected pair.
    expected = load_dataset("tubingen/1")
    dataset = load_dataset("tubingen/1", n_samples=10, seed=42)
    assert dataset.data.shape == (10, expected.data.shape[1])
    assert list(dataset.data.columns) == list(expected.data.columns)
    assert set(dataset.ground_truth.edges()) == set(expected.ground_truth.edges())

    # Simulator-specific kwargs are ignored with a warning.
    with pytest.warns(UserWarning, match="ignore"):
        actual = load_dataset("tubingen/1", edge_prob=0.3)
    pd.testing.assert_frame_equal(actual.data, expected.data)
    assert set(actual.ground_truth.edges()) == set(expected.ground_truth.edges())

def test_load_feedbacks_dataset():
    dataset = load_dataset("feedbacks_network1_amp")
    assert dataset.name == "feedbacks_network1_amp"
    assert dataset.data.shape == (500, 5)
    assert isinstance(dataset.data, pd.DataFrame)
    assert isinstance(dataset.ground_truth, DAG)
    assert set(dataset.ground_truth.nodes()) == set(dataset.data.columns)

    # sim_id selects one of the 60 independent simulation runs; the shared
    # ground-truth graph does not change across runs.
    ds_sim1 = load_dataset("feedbacks_network1_amp", sim_id=1)
    ds_sim2 = load_dataset("feedbacks_network1_amp", sim_id=2)
    assert not ds_sim1.data.equals(ds_sim2.data)
    assert set(ds_sim1.ground_truth.edges()) == set(ds_sim2.ground_truth.edges())

    with pytest.raises(ValueError, match="sim_id"):
        load_dataset("feedbacks_network1_amp", sim_id=61)

    # n_samples truncates to the first n rows of the selected run.
    truncated = load_dataset("feedbacks_network1_amp", sim_id=1, n_samples=10)
    pd.testing.assert_frame_equal(truncated.data, ds_sim1.data.iloc[:10].reset_index(drop=True))

def test_tubingen_missing_data_tag():
    for i in [1, 47, 108]:
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
        list_datasets(is_paraterized=True)

    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_datasets(num_samples=100)


def test_static_dataset_n_samples():
    full = load_dataset("sachs_discrete")

    # n_samples controls the number of returned rows.
    sampled = load_dataset("sachs_discrete", n_samples=100, seed=42)
    assert sampled.data.shape == (100, full.data.shape[1])
    assert list(sampled.data.columns) == list(full.data.columns)

    # Oversized n_samples is capped at the full dataset size and warns.
    with pytest.warns(UserWarning, match="Requested"):
        oversized = load_dataset("sachs_discrete", n_samples=999999)
    assert oversized.data.shape[0] == full.data.shape[0]

    # The same seed gives the same subsample.
    ds1 = load_dataset("sachs_discrete", n_samples=50, seed=42)
    ds2 = load_dataset("sachs_discrete", n_samples=50, seed=42)
    pd.testing.assert_frame_equal(ds1.data, ds2.data)
