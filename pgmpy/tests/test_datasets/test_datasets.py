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


def test_load_tubingen_dataset():

    for i in [1, 47, 108]:
        dataset = load_dataset(f"tubingen/{i}")

        assert dataset.name == f"tubingen/{i}"
        assert isinstance(dataset.data, pd.DataFrame)
        assert list(dataset.data.columns) == ["x", "y"]

        assert isinstance(dataset.ground_truth, DAG)


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
        list_datasets(is_paraterized=True)  # typo

    with pytest.raises(ValueError, match="Unrecognized filter argument"):
        list_datasets(num_samples=100)  # wrong key name entirely


def test_static_dataset_n_samples():
    # n_samples controls the number of returned rows.
    dataset = load_dataset("sachs_discrete", n_samples=100, seed=42)
    assert dataset.data.shape[0] == 100
    assert isinstance(dataset.data, pd.DataFrame)

    # Oversized n_samples is capped at the dataset size and warns.
    full = load_dataset("sachs_discrete")
    with pytest.warns(UserWarning, match="Requested"):
        oversized = load_dataset("sachs_discrete", n_samples=999999)
    assert oversized.data.shape[0] == full.data.shape[0]

    # The same seed gives the same subsample.
    ds1 = load_dataset("sachs_discrete", n_samples=50, seed=42)
    ds2 = load_dataset("sachs_discrete", n_samples=50, seed=42)
    pd.testing.assert_frame_equal(ds1.data.reset_index(drop=True), ds2.data.reset_index(drop=True))


def test_static_dataset_kwargs():
    from pgmpy.datasets.sachs import SachsDiscrete

    # Static dataframes should not accept simulator-specific kwargs.
    with pytest.raises(TypeError):
        load_dataset("sachs_discrete", edge_prob=0.3)

    # Static ground truth ignores forwarded kwargs for load_dataset compatibility.
    gt = SachsDiscrete.load_ground_truth(seed=42, n_nodes=8)
    assert gt is not None
    assert isinstance(gt, DAG)


def test_simulation_mixin_contract():
    from pgmpy.datasets._base import _SimulationMixin

    with pytest.raises(NotImplementedError):
        _SimulationMixin.load_dataframe()

    with pytest.raises(NotImplementedError):
        _SimulationMixin.load_ground_truth()


def test_covariance_n_samples_and_seed():
    # n_samples controls generated covariance dataset size.
    dataset = load_dataset("goldberg", n_samples=25, seed=42)
    assert dataset.data.shape[0] == 25

    # The same seed gives the same generated covariance data.
    ds1 = load_dataset("goldberg", seed=42)
    ds2 = load_dataset("goldberg", seed=42)
    pd.testing.assert_frame_equal(ds1.data, ds2.data)


def test_tubingen_n_samples_and_sim_kwargs():
    # Tubingen supports n_samples by subsampling the selected pair.
    dataset = load_dataset("tubingen/1", n_samples=10, seed=42)
    assert dataset.data.shape[0] == 10
    assert list(dataset.data.columns) == ["x", "y"]
    assert isinstance(dataset.ground_truth, DAG)

    # Simulator-specific kwargs are ignored because Tubingen uses pair_id dispatch.
    with pytest.warns(UserWarning, match="ignore"):
        dataset = load_dataset("tubingen/1", edge_prob=0.3)
    assert isinstance(dataset.data, pd.DataFrame)
