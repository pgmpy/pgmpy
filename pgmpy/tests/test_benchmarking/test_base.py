import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.benchmarking._base import _get_dataset_labels, _resolve_dataset
from pgmpy.datasets._base import Dataset
from pgmpy.datasets.anm import AdditiveNoiseModel
from pgmpy.datasets.linear_gaussian_scm import LinearGaussianSCM

# ---------------------------------------------------------------------------
# _get_dataset_labels
# ---------------------------------------------------------------------------


def test_get_dataset_labels():
    sim = AdditiveNoiseModel(n_nodes=8, seed=1)
    ds = Dataset(name="prepared", data=pd.DataFrame({"a": [1]}))
    df = pd.DataFrame(np.zeros((3, 2)))

    labels = _get_dataset_labels(["sachs_discrete", sim, ds, df])
    assert labels[0] == "sachs_discrete"
    assert labels[1] == " ".join(repr(sim).split())
    assert labels[2] == "prepared"
    assert labels[3] == "DataFrame_3(3x2)"


def test_get_dataset_labels_duplicates():
    sim_a = LinearGaussianSCM(seed=1)
    sim_b = LinearGaussianSCM(seed=1)
    labels = _get_dataset_labels([sim_a, sim_b])
    assert labels[0] != labels[1]
    assert labels[1].endswith("_1")

    labels = _get_dataset_labels(["a", "a", "a_0"])
    assert len(set(labels)) == 3

    labels = _get_dataset_labels(["a", "a", "a_1"])
    assert len(set(labels)) == 3

    df1 = pd.DataFrame(np.zeros((3, 2)))
    df2 = pd.DataFrame(np.zeros((3, 2)))
    ds1 = Dataset(name="same", data=pd.DataFrame({"x": [1]}))
    ds2 = Dataset(name="same", data=pd.DataFrame({"y": [2]}))
    labels = _get_dataset_labels([df1, df2, ds1, ds2])
    assert len(set(labels)) == 4


@pytest.mark.parametrize(
    ("bad_input", "expected_fragment"),
    [
        (42, "int"),
        ([1, 2], "list"),
        (AdditiveNoiseModel, "the class AdditiveNoiseModel"),
    ],
    ids=["int", "list", "uninstantiated_class"],
)
def test_get_dataset_labels_unsupported_type(bad_input, expected_fragment):
    with pytest.raises(TypeError, match=expected_fragment):
        _get_dataset_labels([bad_input])


# ---------------------------------------------------------------------------
# _resolve_dataset — registered names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "n_samples", "expected_rows"),
    [
        ("sachs_discrete", 100, 100),
        ("linear_gaussian", 200, 200),
        ("lead", 1000, 1000),
        ("tubingen/1", None, None),
    ],
    ids=["static", "simulated", "covariance", "tubingen"],
)
def test_resolve_registered_dataset(name, n_samples, expected_rows):
    ds = _resolve_dataset(name, label=name, n_samples=n_samples, seed=42)
    assert isinstance(ds, Dataset)
    if expected_rows is not None:
        assert ds.data.shape[0] == expected_rows

    ds2 = _resolve_dataset(name, label=name, n_samples=n_samples, seed=42)
    assert np.array_equal(ds.data.values, ds2.data.values)


def test_resolve_registered_dataset_errors():
    with pytest.warns(UserWarning, match="Requested"):
        ds = _resolve_dataset("sachs_discrete", label="sachs_discrete", n_samples=999999, seed=42)
    assert ds.data.shape[0] < 999999

    with pytest.raises(ValueError, match="not found"):
        _resolve_dataset("no_such_dataset", label="no_such_dataset", n_samples=100, seed=42)


# ---------------------------------------------------------------------------
# _resolve_dataset — simulator instances
# ---------------------------------------------------------------------------


def test_resolve_simulator_instance():
    sim = AdditiveNoiseModel(n_nodes=8, seed=1)
    ds = _resolve_dataset(sim, label="anm_8", n_samples=50, seed=999)

    assert ds.data.shape == (50, 8)
    assert set(ds.ground_truth.nodes()) == set(ds.data.columns)
    assert ds.tags["n_samples"] == 50
    assert ds.tags["n_variables"] == 8

    df_direct = sim.load_dataframe(n_samples=50)
    assert np.array_equal(ds.data.values, df_direct.values)


# ---------------------------------------------------------------------------
# _resolve_dataset — supplied data (Dataset and DataFrame)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [None, 2], ids=["n_samples_none", "n_samples_set"])
def test_resolve_supplied_data_is_not_resampled(n_samples):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    ds_obj = Dataset(name="mine", data=df)
    resolved = _resolve_dataset(ds_obj, label="mine", n_samples=n_samples, seed=42)
    assert resolved is ds_obj

    resolved_df = _resolve_dataset(df, label="DataFrame_0(3x2)", n_samples=n_samples, seed=42)
    assert resolved_df.data is df
    assert resolved_df.name == "DataFrame_0(3x2)"
    assert resolved_df.ground_truth is None


# ---------------------------------------------------------------------------
# _resolve_dataset — ground_truth override
# ---------------------------------------------------------------------------


def test_ground_truth_override():
    gt = DAG([("a", "b")])

    ds = Dataset(name="no_gt", data=pd.DataFrame({"a": [1], "b": [2]}))
    resolved = _resolve_dataset(ds, label="no_gt", n_samples=None, seed=None, ground_truth={"no_gt": gt})
    assert resolved.ground_truth is gt
    assert ds.ground_truth is None

    df = pd.DataFrame({"a": [1], "b": [2]})
    resolved_df = _resolve_dataset(df, label="my_df", n_samples=None, seed=None, ground_truth={"my_df": gt})
    assert resolved_df.ground_truth is gt

    resolved_lead = _resolve_dataset("lead", label="lead", n_samples=100, seed=42, ground_truth={"lead": gt})
    assert resolved_lead.ground_truth is gt

    sim = AdditiveNoiseModel(n_nodes=5, seed=1)
    resolved_sim = _resolve_dataset(sim, label="anm", n_samples=50, seed=None, ground_truth={"anm": gt})
    assert resolved_sim.ground_truth is not gt

    resolved_miss = _resolve_dataset(ds, label="no_gt", n_samples=None, seed=None, ground_truth={"other_key": gt})
    assert resolved_miss.ground_truth is None
