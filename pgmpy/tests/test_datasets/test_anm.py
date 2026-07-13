"""Tests for the AdditiveNoiseModel (ANM) simulator dataset."""

import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import DAG
from pgmpy.datasets import list_datasets, load_dataset


def test_load_anm_dataset():
    ds = load_dataset("anm", seed=42)
    assert ds.data.shape == (1000, 5)
    assert isinstance(ds.ground_truth, DAG)
    assert ds.tags["n_variables"] == 5
    assert ds.tags["n_samples"] == 1000
    assert ds.tags["is_simulated"] is True
    assert ds.tags["has_ground_truth"] is True
    assert ds.tags["is_continuous"] is True

    ds2 = load_dataset("anm", seed=42, n_nodes=8, n_samples=200)
    assert ds2.data.shape == (200, 8)
    assert ds2.tags["n_variables"] == 8
    assert ds2.tags["n_samples"] == 200

    # Reproducibility: same seed produces identical data and edges.
    ds_a = load_dataset("anm", seed=99)
    ds_b = load_dataset("anm", seed=99)
    assert list(ds_a.data.columns) == list(ds_b.data.columns)
    assert np.allclose(ds_a.data.values, ds_b.data.values)
    assert set(ds_a.ground_truth.edges()) == set(ds_b.ground_truth.edges())

    # Different function_type produces same DAG but different data values.
    ds_sin = load_dataset("anm", seed=42, function_type=(np.sin,))
    ds_cos = load_dataset("anm", seed=42, function_type=(np.cos,))
    assert set(ds_sin.ground_truth.edges()) == set(ds_cos.ground_truth.edges())
    assert not np.allclose(ds_sin.data.values, ds_cos.data.values)

    # Custom callable integrates without error.
    ds_fn = load_dataset("anm", seed=42, function_type=(lambda x: x**2,))
    assert ds_fn.data.shape == (1000, 5)

    # Edge cases: empty graph and single node.
    ds_empty = load_dataset("anm", seed=42, edge_prob=0)
    assert len(ds_empty.ground_truth.edges()) == 0

    single = DAG()
    single.add_node("X")
    ds_single = load_dataset("anm", seed=42, dag=single)
    assert ds_single.data.shape == (1000, 1)

    # User-provided DAG overrides n_nodes with a warning.
    custom_dag = DAG([("A", "B"), ("B", "C")])
    with pytest.warns(UserWarning, match="dag was provided"):
        ds_dag = load_dataset("anm", seed=42, dag=custom_dag, n_nodes=10)
    assert set(ds_dag.data.columns) == {"A", "B", "C"}

    # Fixed weight uses the same coefficient for every edge.
    ds_fixed = load_dataset("anm", seed=42, weight=1.0)
    assert ds_fixed.data.shape == (1000, 5)

    # Sampled weight range affects data magnitude.
    ds_narrow = load_dataset("anm", seed=42, weight=(-0.01, 0.01))
    ds_wide = load_dataset("anm", seed=42, weight=(-10, 10))
    assert ds_narrow.data.std().mean() < ds_wide.data.std().mean()

    assert "anm" in list_datasets(is_simulated=True)


def test_anm_validation():
    with pytest.raises(TypeError, match="dag must be"):
        load_dataset("anm", seed=42, dag="not_a_dag")

    with pytest.raises(TypeError, match=".sample.*or.*\\.rvs.*method"):
        load_dataset("anm", seed=42, noise="not_a_dist")


def test_anm_scipy_noise():
    from scipy.stats import laplace

    # Scipy noise is reproducible with the same seed.
    ds1 = load_dataset("anm", seed=42, noise=laplace(loc=0, scale=1))
    ds2 = load_dataset("anm", seed=42, noise=laplace(loc=0, scale=1))
    assert ds1.data.shape == (1000, 5)
    assert np.allclose(ds1.data.values, ds2.data.values)


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"),
    reason="skpro not installed",
)
def test_anm_skpro_noise():
    from skpro.distributions import Laplace

    ds = load_dataset("anm", seed=42, noise=Laplace(mu=0, scale=1))
    assert ds.data.shape == (1000, 5)

    # skpro noise is reproducible with the same seed despite skpro's lack of a random_state argument.
    ds_a = load_dataset("anm", seed=42, noise=Laplace(mu=0, scale=1))
    ds_b = load_dataset("anm", seed=42, noise=Laplace(mu=0, scale=1))
    assert np.allclose(ds_a.data.values, ds_b.data.values)

    # Isolated nodes must get independent noise draws from skpro.
    iso_dag = DAG()
    iso_dag.add_node("A")
    iso_dag.add_node("B")
    ds_iso = load_dataset("anm", seed=42, dag=iso_dag, noise=Laplace(mu=0, scale=1))
    assert not np.allclose(ds_iso.data["A"].values, ds_iso.data["B"].values)
