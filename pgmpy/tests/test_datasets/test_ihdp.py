"""Tests for the IHDP semi-synthetic dataset simulator."""

import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import DAG
from pgmpy.datasets import list_datasets, load_dataset


def test_load_ihdp_dataset():
    # Default (Setting B) — shape, types, ground truth.
    ds = load_dataset("ihdp", seed=42)
    assert ds.data.shape == (747, 30)
    assert isinstance(ds.ground_truth, DAG)
    assert ds.data["treatment"].sum() == 139

    # Covariates + treatment are fixed across seeds.
    ds2 = load_dataset("ihdp", seed=99)
    np.testing.assert_array_equal(ds.data["treatment"], ds2.data["treatment"])
    for col in [f"x{i}" for i in range(1, 26)]:
        np.testing.assert_allclose(ds.data[col], ds2.data[col])

    # Outcomes differ across seeds (different beta + noise).
    assert not np.allclose(ds.data["y_factual"], ds2.data["y_factual"])

    # Reproducibility: same seed produces identical data.
    ds_a = load_dataset("ihdp", seed=42)
    ds_b = load_dataset("ihdp", seed=42)
    assert np.allclose(ds_a.data["y_factual"].values, ds_b.data["y_factual"].values)

    # Setting A — homogeneous treatment effect.
    ds_a_set = load_dataset("ihdp", seed=42, setting="A")
    ite_a = ds_a_set.data["mu1"] - ds_a_set.data["mu0"]
    np.testing.assert_allclose(ite_a, 4.0, atol=1e-10)

    # Setting B — heterogeneous treatment effect, but ATT-on-treated
    # is calibrated exactly.
    ite_b = ds.data["mu1"] - ds.data["mu0"]
    assert np.std(ite_b) > 0
    att_b = ite_b[ds.data["treatment"] == 1].mean()
    np.testing.assert_allclose(att_b, 4.0, atol=1e-8)

    # Custom omega — Setting A (literal constant effect).
    ds_omega = load_dataset("ihdp", seed=42, setting="A", omega=10.0)
    ite_omega = ds_omega.data["mu1"] - ds_omega.data["mu0"]
    np.testing.assert_allclose(ite_omega, 10.0, atol=1e-10)

    # Custom omega — Setting B (calibrated ATT-on-treated).
    ds_omega_b = load_dataset("ihdp", seed=42, setting="B", omega=10.0)
    att_omega_b = (ds_omega_b.data["mu1"] - ds_omega_b.data["mu0"])[ds_omega_b.data["treatment"] == 1].mean()
    np.testing.assert_allclose(att_omega_b, 10.0, atol=1e-8)

    # n_samples warning.
    with pytest.warns(UserWarning, match="n_samples is ignored"):
        ds_n = load_dataset("ihdp", seed=42, n_samples=100)
    assert ds_n.data.shape[0] == 747

    # Ground truth DAG has roles.
    gt = ds.ground_truth
    assert set(gt.get_role("exposures")) == {"treatment"}
    assert set(gt.get_role("outcomes")) == {"y_factual"}
    assert len(gt.get_role("adjustment")) == 25

    # Discoverable.
    assert "ihdp" in list_datasets(is_simulated=True)


def test_ihdp_dataset_tags():
    ds = load_dataset("ihdp", seed=42)
    assert ds.tags["n_variables"] == 30
    assert ds.tags["n_samples"] == 747
    assert ds.tags["is_simulated"] is True
    assert ds.tags["has_ground_truth"] is True
    assert ds.tags["is_continuous"] is True
    assert ds.tags["is_mixed"] is True


def test_ihdp_validation():
    with pytest.raises(ValueError, match="Unknown setting"):
        load_dataset("ihdp", seed=42, setting="C")

    with pytest.raises(TypeError, match=".sample.*or.*\\.rvs.*method"):
        load_dataset("ihdp", seed=42, noise="not_a_dist")


def test_ihdp_scipy_noise():
    from scipy.stats import laplace

    ds = load_dataset("ihdp", seed=42, noise=laplace(loc=0, scale=1))
    assert ds.data.shape == (747, 30)


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"),
    reason="skpro not installed",
)
def test_ihdp_skpro_noise():
    from skpro.distributions import Laplace

    ds = load_dataset("ihdp", seed=42, noise=Laplace(mu=0, scale=1))
    assert ds.data.shape == (747, 30)
