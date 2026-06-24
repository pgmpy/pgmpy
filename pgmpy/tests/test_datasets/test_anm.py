import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import DAG
from pgmpy.datasets import list_datasets, load_dataset


def test_load_anm_dataset():
    # Default parameters — shape, types, ground truth.
    ds = load_dataset("anm", seed=42)
    assert ds.data.shape == (1000, 5)
    assert isinstance(ds.ground_truth, DAG)
    assert set(ds.ground_truth.nodes()) == set(ds.data.columns)

    # Custom n_nodes / n_samples.
    ds_custom = load_dataset("anm", n_samples=200, seed=42, n_nodes=8, edge_prob=0.3)
    assert ds_custom.data.shape == (200, 8)
    assert set(ds_custom.data.columns) == set(ds_custom.ground_truth.nodes())

    # Seed reproducibility — same seed produces identical data and edges.
    ds_repeat = load_dataset("anm", seed=42)
    pd.testing.assert_frame_equal(ds.data, ds_repeat.data)
    assert set(ds.ground_truth.edges()) == set(ds_repeat.ground_truth.edges())

    # Different function_type : different data, same structure.
    ds_poly = load_dataset("anm", seed=42, function_type="polynomial")
    assert not ds.data.equals(ds_poly.data)
    assert set(ds.ground_truth.edges()) == set(ds_poly.ground_truth.edges())

    # All three presets produce valid output.
    for ft in ["sine_add", "polynomial", "sigmoid_add"]:
        ds_ft = load_dataset("anm", seed=42, function_type=ft)
        assert ds_ft.data.shape == (1000, 5)

    # edge_prob=0 : all isolated nodes, no edges.
    ds_empty = load_dataset("anm", seed=42, n_nodes=8, edge_prob=0)
    assert set(ds_empty.ground_truth.nodes()) == {f"X_{i}" for i in range(8)}
    assert not ds_empty.ground_truth.edges()

    # User-specified DAG overrides n_nodes / edge_prob with a warning.
    custom_dag = DAG([("A", "B"), ("B", "C")])
    with pytest.warns(UserWarning, match="ignored"):
        ds_dag = load_dataset("anm", seed=42, dag=custom_dag, n_nodes=99)
    assert set(ds_dag.data.columns) == {"A", "B", "C"}
    assert set(ds_dag.ground_truth.edges()) == set(custom_dag.edges())

    # Custom callable function_type.
    ds_fn = load_dataset("anm", seed=42, function_type=lambda x: np.tanh(x).sum(axis=1))
    assert ds_fn.data.shape == (1000, 5)

    # Single-node DAG (no edges, just noise).
    single_dag = DAG()
    single_dag.add_node("X")
    ds_single = load_dataset("anm", seed=42, dag=single_dag)
    assert ds_single.data.shape == (1000, 1)
    assert list(ds_single.data.columns) == ["X"]

    # noise_scale controls noise magnitude.
    ds_low = load_dataset("anm", seed=42, noise_scale=0.01)
    ds_high = load_dataset("anm", seed=42, noise_scale=10.0)
    assert ds_low.data.std().mean() < ds_high.data.std().mean()

    # Discoverable via list_datasets.
    assert "anm" in list_datasets(is_simulated=True)


def test_anm_validation():
    # Invalid function_type string.
    with pytest.raises(ValueError, match="Unknown function_type"):
        load_dataset("anm", seed=42, function_type="invalid_type")

    # function_type is neither string nor callable.
    with pytest.raises(TypeError, match="function_type must be"):
        load_dataset("anm", seed=42, function_type=42)

    # Invalid weight_range (lower > upper).
    with pytest.raises(ValueError, match="weight_range"):
        load_dataset("anm", seed=42, weight_range=(2.0, 0.5))

    # Invalid weight_range (non-positive).
    with pytest.raises(ValueError, match="weight_range"):
        load_dataset("anm", seed=42, weight_range=(-1.0, 2.0))

    # Invalid weight_range (wrong length).
    with pytest.raises(ValueError, match="weight_range"):
        load_dataset("anm", seed=42, weight_range=(1.0,))

    # n_nodes < 1.
    with pytest.raises(ValueError, match="n_nodes"):
        load_dataset("anm", seed=42, n_nodes=0)

    # edge_prob out of range.
    with pytest.raises(ValueError, match="edge_prob"):
        load_dataset("anm", seed=42, edge_prob=1.5)

    # noise_scale non-positive.
    with pytest.raises(ValueError, match="noise_scale"):
        load_dataset("anm", seed=42, noise_scale=-1.0)

    # n_samples non-positive.
    with pytest.raises(ValueError, match="n_samples"):
        load_dataset("anm", seed=42, n_samples=0)


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"),
    reason="skpro not installed",
)
def test_anm_skpro_noise():
    from skpro.distributions import Laplace

    # Custom skpro noise produces valid data.
    ds = load_dataset("anm", seed=42, noise=Laplace(mu=0, scale=1))
    assert ds.data.shape == (1000, 5)

    # noise_scale with custom noise raises ValueError.
    with pytest.raises(ValueError, match="noise_scale cannot be used"):
        load_dataset("anm", seed=42, noise=Laplace(mu=0, scale=1), noise_scale=2.0)

    # Custom noise should be sampled independently for each node.
    iso_dag = DAG()
    iso_dag.add_node("A")
    iso_dag.add_node("B")
    ds_iso = load_dataset("anm", seed=42, dag=iso_dag, noise=Laplace(mu=0, scale=1))
    assert not np.allclose(ds_iso.data["A"].values, ds_iso.data["B"].values), (
        "Noise draws for isolated nodes A and B should be independent"
    )
