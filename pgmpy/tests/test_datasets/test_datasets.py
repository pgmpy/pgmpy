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


# --- Subsampling tests ---


def test_subsampling_static_dataset():
    dataset = load_dataset("sachs_continuous", n_samples=100, seed=42)
    assert dataset.data.shape[0] == 100
    assert isinstance(dataset.data, pd.DataFrame)


def test_subsampling_caps_at_dataset_size():
    full = load_dataset("sachs_continuous")
    subsampled = load_dataset("sachs_continuous", n_samples=999999)
    assert subsampled.data.shape[0] == full.data.shape[0]


def test_subsampling_seed_reproducibility():
    ds1 = load_dataset("sachs_continuous", n_samples=50, seed=42)
    ds2 = load_dataset("sachs_continuous", n_samples=50, seed=42)
    pd.testing.assert_frame_equal(ds1.data.reset_index(drop=True), ds2.data.reset_index(drop=True))


def test_static_dataset_rejects_sim_kwargs():
    with pytest.raises(TypeError):
        load_dataset("sachs_continuous", edge_prob=0.3)


def test_static_ground_truth_ignores_kwargs():
    from pgmpy.datasets.sachs import SachsContinuous

    # Should not raise even though seed and sim_kwargs are forwarded
    gt = SachsContinuous.load_ground_truth(seed=42, n_nodes=8)
    assert gt is not None
    assert isinstance(gt, DAG)


# --- _SimulationMixin contract tests ---


def test_simulation_mixin_not_implemented():
    from pgmpy.datasets._base import _SimulationMixin

    with pytest.raises(NotImplementedError):
        _SimulationMixin.load_dataframe()

    with pytest.raises(NotImplementedError):
        _SimulationMixin.load_ground_truth()


# --- Covariance backward compatibility ---


def test_covariance_dataset_still_works():
    for name in ["goldberg", "spartina"]:
        dataset = load_dataset(name)
        assert dataset.name == name
        assert isinstance(dataset.data, pd.DataFrame)
        assert dataset.data.shape[0] > 0


# --- LinearGaussianSCM tests ---


def test_load_linear_gaussian_scm():
    ds = load_dataset("linear_gaussian_scm", n_samples=500, seed=42, n_nodes=8)
    assert ds.data.shape == (500, 8)
    assert isinstance(ds.data, pd.DataFrame)
    assert isinstance(ds.ground_truth, DAG)
    assert ds.tags["is_simulated"] is True
    assert ds.tags["has_ground_truth"] is True


def test_lgscm_sim_kwargs():
    ds = load_dataset("linear_gaussian_scm", n_samples=200, seed=7, n_nodes=6)
    assert ds.data.shape[1] == 6


def test_lgscm_seed_reproducibility():
    ds1 = load_dataset("linear_gaussian_scm", n_samples=300, seed=42, n_nodes=5)
    ds2 = load_dataset("linear_gaussian_scm", n_samples=300, seed=42, n_nodes=5)
    pd.testing.assert_frame_equal(ds1.data, ds2.data)
    assert set(ds1.ground_truth.edges()) == set(ds2.ground_truth.edges())


def test_lgscm_different_seed():
    ds1 = load_dataset("linear_gaussian_scm", n_samples=300, seed=42, n_nodes=5)
    ds2 = load_dataset("linear_gaussian_scm", n_samples=300, seed=123, n_nodes=5)
    # Data should differ; don't assert edge differences — small graphs can match by chance
    assert not ds1.data.equals(ds2.data)


def test_lgscm_sim_params_tag():
    from pgmpy.datasets.linear_gaussian_scm import LinearGaussianSCM

    sim_params = LinearGaussianSCM.get_class_tag("sim_params")
    assert sim_params is not None
    assert "n_nodes" in sim_params
    assert "edge_prob" in sim_params
    assert "noise_scale" in sim_params


def test_lgscm_in_list_datasets():
    assert "linear_gaussian_scm" in list_datasets(is_simulated=True)


def test_lgscm_isolated_nodes_preserved():
    ds = load_dataset("linear_gaussian_scm", n_samples=100, seed=42, n_nodes=8, edge_prob=0.0)
    assert len(ds.ground_truth.nodes()) == 8
    assert len(ds.ground_truth.edges()) == 0


def test_tubingen_rejects_extra_args():
    with pytest.raises(ValueError, match="do not support"):
        load_dataset("tubingen/1", n_samples=10)
    with pytest.raises(ValueError, match="do not support"):
        load_dataset("tubingen/1", seed=42)
    with pytest.raises(ValueError, match="do not support"):
        load_dataset("tubingen/1", edge_prob=0.3)
