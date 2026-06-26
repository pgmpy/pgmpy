"""Tests for the AdditiveNoiseModel (ANM) simulator dataset."""

import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import DAG
from pgmpy.datasets import load_dataset


def test_load_anm_dataset():
    ds = load_dataset("anm", seed=42)
    assert ds.data.shape == (1000, 5)
    assert isinstance(ds.ground_truth, DAG)

    ds2 = load_dataset("anm", seed=42, n_nodes=8, n_samples=200)
    assert ds2.data.shape == (200, 8)

    # Reproducibility: same seed → identical data and edges.
    ds_a = load_dataset("anm", seed=99)
    ds_b = load_dataset("anm", seed=99)
    assert list(ds_a.data.columns) == list(ds_b.data.columns)
    assert np.allclose(ds_a.data.values, ds_b.data.values)
    assert set(ds_a.ground_truth.edges()) == set(ds_b.ground_truth.edges())

    # Different function_type → same DAG but different data values.
    ds_sin = load_dataset("anm", seed=42, function_type={np.sin})
    ds_cos = load_dataset("anm", seed=42, function_type={np.cos})
    assert set(ds_sin.ground_truth.edges()) == set(ds_cos.ground_truth.edges())
    assert not np.allclose(ds_sin.data.values, ds_cos.data.values)

    # Custom callable integrates without error.
    ds_fn = load_dataset("anm", seed=42, function_type={lambda x: x**2})
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

    from pgmpy.datasets import list_datasets

    assert "anm" in list_datasets(is_simulated=True)


def test_anm_validation():
    with pytest.raises(TypeError, match="function_type must be"):
        load_dataset("anm", seed=42, function_type="sine")

    with pytest.raises(TypeError, match="All items.*callable"):
        load_dataset("anm", seed=42, function_type={42})

    with pytest.raises(ValueError, match="at least one"):
        load_dataset("anm", seed=42, function_type=set())

    with pytest.raises(ValueError, match="weight_range"):
        load_dataset("anm", seed=42, weight_range=(2.0, -2.0))

    with pytest.raises(ValueError, match="weight_range"):
        load_dataset("anm", seed=42, weight_range=(1.0,))

    with pytest.raises(ValueError, match="n_nodes"):
        load_dataset("anm", seed=42, n_nodes=0)

    with pytest.raises(ValueError, match="edge_prob"):
        load_dataset("anm", seed=42, edge_prob=1.5)

    with pytest.raises(ValueError, match="n_samples"):
        load_dataset("anm", seed=42, n_samples=0)

    with pytest.raises(TypeError, match="dag must be"):
        load_dataset("anm", seed=42, dag="not_a_dag")


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"),
    reason="skpro not installed",
)
def test_anm_skpro_noise():
    from skpro.distributions import Laplace

    ds = load_dataset("anm", seed=42, noise=Laplace(mu=0, scale=1))
    assert ds.data.shape == (1000, 5)

    # Isolated nodes must get independent noise draws from skpro.
    iso_dag = DAG()
    iso_dag.add_node("A")
    iso_dag.add_node("B")
    ds_iso = load_dataset("anm", seed=42, dag=iso_dag, noise=Laplace(mu=0, scale=1))
    assert not np.allclose(ds_iso.data["A"].values, ds_iso.data["B"].values)

    with pytest.raises(TypeError, match="noise must be a skpro"):
        load_dataset("anm", seed=42, noise="not_a_dist")
