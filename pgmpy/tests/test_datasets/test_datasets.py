import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.datasets import list_datasets, load_dataset
from pgmpy.datasets.ihdp import IHDP, IHDP2
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
    "ihdp",
    "ihdp2",
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
    assert "ihdp" in found_datasets
    assert "ihdp2" in found_datasets

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


@pytest.mark.parametrize(
    "dataset_class,dataset_name", [(IHDP, "ihdp"), (IHDP2, "ihdp2")]
)
def test_load_ihdp_family(monkeypatch, dataset_class, dataset_name):
    n_samples = dataset_class.get_class_tag("n_samples")
    n_covariates = 25
    values = np.zeros((n_samples, 5 + n_covariates), dtype=float)
    values[:, 0] = np.arange(n_samples) % 2
    values[:, 1] = 2.0
    values[:, 2] = 1.0
    values[:, 3] = 0.5
    values[:, 4] = 1.5
    values[:, 5:] = np.arange(1, n_covariates + 1)
    raw_data = pd.DataFrame(values).to_csv(index=False, header=False).encode("utf-8")

    def mock_get_raw_data(cls, data_type, url):
        assert data_type == "data"
        assert url == dataset_class.data_url
        return raw_data

    monkeypatch.setattr(dataset_class, "_get_raw_data", classmethod(mock_get_raw_data))
    dataset = load_dataset(dataset_name)
    assert dataset.data.shape == (
        dataset.tags["n_samples"],
        dataset.tags["n_variables"],
    )
    expected_columns = {
        "treatment",
        "y_factual",
        "y_cfactual",
        "mu0",
        "mu1",
    }
    assert expected_columns.issubset(set(dataset.data.columns))
    assert dataset.data["treatment"].dtype.name == "category"
    assert dataset.ground_truth is None
    assert dataset.expert_knowledge is None


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
