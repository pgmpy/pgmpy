import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


@pytest.fixture(autouse=True)
def shutdown_executor():
    yield
    get_reusable_executor().shutdown(wait=True)


requires_daft = pytest.mark.skipif(
    not _check_soft_dependencies("daft-pgm", severity="none"),
    reason="execute only if required dependency present",
)

requires_torch = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="requires torch to be installed",
)


@pytest.fixture
def models():
    m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
    model_latent = DiscreteBayesianNetwork([("A", "C"), ("B", "C")], latents=["C"])
    dag_with_latents = DAG([("A", "B"), ("B", "C")], latents=["C"])
    d1 = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
    d2 = pd.DataFrame(
        data={
            "A": [0, 0, 1, 0, 2, 0, 2, 1, 0, 2],
            "B": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "C": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        }
    )

    est1 = BayesianEstimator(m1, d1)
    est2 = BayesianEstimator(m1, d1, state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]})
    est3 = BayesianEstimator(m1, d2)
    return {
        "model_latent": model_latent,
        "dag_with_latents": dag_with_latents,
        "d1": d1,
        "est1": est1,
        "est2": est2,
        "est3": est3,
    }


def test_error_latent_model(models):
    with pytest.raises(ValueError):
        BayesianEstimator(models["model_latent"], models["d1"])
    with pytest.raises(ValueError):
        BayesianEstimator(models["dag_with_latents"], models["d1"])


def test_estimate_cpd_dirichlet(models):
    est1 = models["est1"]

    cpd_A = est1.estimate_cpd("A", prior_type="dirichlet", pseudo_counts=[[0], [1]])
    cpd_A_exp = TabularCPD(
        variable="A",
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={"A": [0, 1]},
    )

    assert cpd_A == cpd_A_exp

    pseudo_counts = np.array([[0], [1]])
    cpd_A = est1.estimate_cpd("A", prior_type="dirichlet", pseudo_counts=pseudo_counts)
    assert cpd_A == cpd_A_exp

    cpd_B = est1.estimate_cpd("B", prior_type="dirichlet", pseudo_counts=[[9], [3]])
    cpd_B_exp = TabularCPD("B", 2, [[11.0 / 15], [4.0 / 15]], state_names={"B": [0, 1]})
    assert cpd_B == cpd_B_exp

    cpd_C = est1.estimate_cpd(
        "C",
        prior_type="dirichlet",
        pseudo_counts=[[0.4, 0.4, 0.4, 0.4], [0.6, 0.6, 0.6, 0.6]],
    )
    cpd_C_exp = TabularCPD(
        "C",
        2,
        [[0.2, 0.2, 0.7, 0.4], [0.8, 0.8, 0.3, 0.6]],
        evidence=["A", "B"],
        evidence_card=[2, 2],
        state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
    )
    assert cpd_C == cpd_C_exp


def test_estimate_cpd_improper_prior(models):
    cpd_C = models["est1"].estimate_cpd("C", prior_type="dirichlet", pseudo_counts=[[0, 0, 0, 0], [0, 0, 0, 0]])
    cpd_C_correct = TabularCPD(
        "C",
        2,
        [[0.0, 0.0, 1.0, np.nan], [1.0, 1.0, 0.0, np.nan]],
        evidence=["A", "B"],
        evidence_card=[2, 2],
        state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
    )

    assert ((cpd_C.values == cpd_C_correct.values) | np.isnan(cpd_C.values) & np.isnan(cpd_C_correct.values)).all()


def test_estimate_cpd_shortcuts(models):
    est2, est3 = models["est2"], models["est3"]

    cpd_C1 = est2.estimate_cpd("C", prior_type="BDeu", equivalent_sample_size=9)
    cpd_C1_correct = TabularCPD(
        "C",
        3,
        [
            [0.2, 0.2, 0.6, 1.0 / 3, 1.0 / 3, 1.0 / 3],
            [0.6, 0.6, 0.2, 1.0 / 3, 1.0 / 3, 1.0 / 3],
            [0.2, 0.2, 0.2, 1.0 / 3, 1.0 / 3, 1.0 / 3],
        ],
        evidence=["A", "B"],
        evidence_card=[3, 2],
        state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]},
    )
    assert cpd_C1 == cpd_C1_correct

    cpd_C2 = est3.estimate_cpd("C", prior_type="K2")
    cpd_C2_correct = TabularCPD(
        "C",
        2,
        [
            [0.5, 0.6, 1.0 / 3, 2.0 / 3, 0.75, 2.0 / 3],
            [0.5, 0.4, 2.0 / 3, 1.0 / 3, 0.25, 1.0 / 3],
        ],
        evidence=["A", "B"],
        evidence_card=[3, 2],
        state_names={"A": [0, 1, 2], "B": ["X", "Y"], "C": [0, 1]},
    )

    assert cpd_C2 == cpd_C2_correct


def test_get_parameters(models):
    est3 = models["est3"]
    cpds = {
        est3.estimate_cpd("A"),
        est3.estimate_cpd("B"),
        est3.estimate_cpd("C"),
    }

    assert set(est3.get_parameters(n_jobs=1)) == cpds


def test_get_parameters2(models):
    est3 = models["est3"]
    pseudo_counts = {
        "A": [[1], [2], [3]],
        "B": [[4], [5]],
        "C": [[6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7]],
    }

    cpds = {
        est3.estimate_cpd("A", prior_type="dirichlet", pseudo_counts=pseudo_counts["A"]),
        est3.estimate_cpd("B", prior_type="dirichlet", pseudo_counts=pseudo_counts["B"]),
        est3.estimate_cpd("C", prior_type="dirichlet", pseudo_counts=pseudo_counts["C"]),
    }

    assert set(est3.get_parameters(prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1)) == cpds


def test_get_parameters3(models):
    est3 = models["est3"]
    pseudo_counts = 0.1
    cpds = {
        est3.estimate_cpd("A", prior_type="dirichlet", pseudo_counts=pseudo_counts),
        est3.estimate_cpd("B", prior_type="dirichlet", pseudo_counts=pseudo_counts),
        est3.estimate_cpd("C", prior_type="dirichlet", pseudo_counts=pseudo_counts),
    }
    assert set(est3.get_parameters(prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1)) == cpds


def test_node_specific_equivalent_sample_size(models):
    est3 = models["est3"]
    ess_dict = {"A": 10, "B": 20, "C": 15}
    cpds_dict = est3.get_parameters(prior_type="bdeu", equivalent_sample_size=ess_dict, n_jobs=1)
    cpds_manual = {
        est3.estimate_cpd("A", prior_type="bdeu", equivalent_sample_size=10),
        est3.estimate_cpd("B", prior_type="bdeu", equivalent_sample_size=20),
        est3.estimate_cpd("C", prior_type="bdeu", equivalent_sample_size=15),
    }
    assert set(cpds_dict) == cpds_manual


def test_node_specific_ess_partial_dict(models):
    est3 = models["est3"]
    """Test that unspecified nodes default to 0 (or equivalent behavior) when dict is partial."""
    ess_dict = {"A": 10, "C": 15}
    cpd_A_dict = est3.estimate_cpd("A", prior_type="bdeu", equivalent_sample_size=ess_dict)
    cpd_B_dict = est3.estimate_cpd("B", prior_type="bdeu", equivalent_sample_size=ess_dict)
    cpd_C_dict = est3.estimate_cpd("C", prior_type="bdeu", equivalent_sample_size=ess_dict)

    cpd_A_manual = est3.estimate_cpd("A", prior_type="bdeu", equivalent_sample_size=10)
    cpd_C_manual = est3.estimate_cpd("C", prior_type="bdeu", equivalent_sample_size=15)
    cpd_B_manual = est3.estimate_cpd("B", prior_type="bdeu", equivalent_sample_size=0)

    assert cpd_A_dict == cpd_A_manual
    assert cpd_B_dict == cpd_B_manual
    assert cpd_C_dict == cpd_C_manual


def test_node_specific_ess_matches_uniform_ess(models):
    est3 = models["est3"]
    """Test that uniform ESS dict matches scalar ESS."""
    ess_value = 12
    ess_dict = {"A": ess_value, "B": ess_value, "C": ess_value}
    cpds_scalar = est3.get_parameters(prior_type="bdeu", equivalent_sample_size=ess_value, n_jobs=1)
    cpds_dict = est3.get_parameters(prior_type="bdeu", equivalent_sample_size=ess_dict, n_jobs=1)
    assert set(cpds_scalar) == set(cpds_dict)


@pytest.fixture
def torch_models():
    config.set_backend("torch")
    m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
    model_latent = DiscreteBayesianNetwork([("A", "C"), ("B", "C")], latents=["C"])
    d1 = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
    d2 = pd.DataFrame(
        data={
            "A": [0, 0, 1, 0, 2, 0, 2, 1, 0, 2],
            "B": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "C": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    est1 = BayesianEstimator(m1, d1)
    est2 = BayesianEstimator(m1, d1, state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]})
    est3 = BayesianEstimator(m1, d2)
    yield {
        "model_latent": model_latent,
        "d1": d1,
        "est1": est1,
        "est2": est2,
        "est3": est3,
    }
    config.set_backend("numpy")


@requires_torch
def test_error_latent_model_torch(torch_models):
    with pytest.raises(ValueError):
        BayesianEstimator(torch_models["model_latent"], torch_models["d1"])


@requires_daft
@requires_torch
def test_estimate_cpd_dirichlet_torch(torch_models):
    est1 = torch_models["est1"]
    cpd_A = est1.estimate_cpd("A", prior_type="dirichlet", pseudo_counts=[[0], [1]])
    cpd_A_exp = TabularCPD(
        variable="A",
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={"A": [0, 1]},
    )
    assert cpd_A == cpd_A_exp

    # also test passing pseudo_counts as np.array
    pseudo_counts = np.array([[0], [1]])
    cpd_A = est1.estimate_cpd("A", prior_type="dirichlet", pseudo_counts=pseudo_counts)
    assert cpd_A == cpd_A_exp

    cpd_B = est1.estimate_cpd("B", prior_type="dirichlet", pseudo_counts=[[9], [3]])
    cpd_B_exp = TabularCPD("B", 2, [[11.0 / 15], [4.0 / 15]], state_names={"B": [0, 1]})
    assert cpd_B == cpd_B_exp

    cpd_C = est1.estimate_cpd(
        "C",
        prior_type="dirichlet",
        pseudo_counts=[[0.4, 0.4, 0.4, 0.4], [0.6, 0.6, 0.6, 0.6]],
    )
    cpd_C_exp = TabularCPD(
        "C",
        2,
        [[0.2, 0.2, 0.7, 0.4], [0.8, 0.8, 0.3, 0.6]],
        evidence=["A", "B"],
        evidence_card=[2, 2],
        state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
    )
    assert cpd_C == cpd_C_exp


@requires_torch
def test_estimate_cpd_improper_prior_torch(torch_models):
    cpd_C = torch_models["est1"].estimate_cpd("C", prior_type="dirichlet", pseudo_counts=[[0, 0, 0, 0], [0, 0, 0, 0]])
    cpd_C_correct = TabularCPD(
        "C",
        2,
        [[0.0, 0.0, 1.0, np.nan], [1.0, 1.0, 0.0, np.nan]],
        evidence=["A", "B"],
        evidence_card=[2, 2],
        state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
    )
    backend = config.get_compute_backend()
    # manual comparison because np.nan != np.nan
    assert (
        (cpd_C.values == cpd_C_correct.values) | backend.isnan(cpd_C.values) & backend.isnan(cpd_C_correct.values)
    ).all()


@requires_daft
@requires_torch
def test_estimate_cpd_shortcuts_torch(torch_models):
    est2, est3 = torch_models["est2"], torch_models["est3"]
    cpd_C1 = est2.estimate_cpd("C", prior_type="BDeu", equivalent_sample_size=9)
    cpd_C1_correct = TabularCPD(
        "C",
        3,
        [
            [0.2, 0.2, 0.6, 1.0 / 3, 1.0 / 3, 1.0 / 3],
            [0.6, 0.6, 0.2, 1.0 / 3, 1.0 / 3, 1.0 / 3],
            [0.2, 0.2, 0.2, 1.0 / 3, 1.0 / 3, 1.0 / 3],
        ],
        evidence=["A", "B"],
        evidence_card=[3, 2],
        state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]},
    )
    assert cpd_C1 == cpd_C1_correct

    cpd_C2 = est3.estimate_cpd("C", prior_type="K2")
    cpd_C2_correct = TabularCPD(
        "C",
        2,
        [
            [0.5, 0.6, 1.0 / 3, 2.0 / 3, 0.75, 2.0 / 3],
            [0.5, 0.4, 2.0 / 3, 1.0 / 3, 0.25, 1.0 / 3],
        ],
        evidence=["A", "B"],
        evidence_card=[3, 2],
        state_names={"A": [0, 1, 2], "B": ["X", "Y"], "C": [0, 1]},
    )
    assert cpd_C2 == cpd_C2_correct


@requires_daft
@requires_torch
def test_get_parameters_torch(torch_models):
    est3 = torch_models["est3"]
    cpds = [
        est3.estimate_cpd("A"),
        est3.estimate_cpd("B"),
        est3.estimate_cpd("C"),
    ]
    all_cpds = est3.get_parameters(n_jobs=1)
    assert sorted(cpds, key=lambda t: t.variables[0]) == sorted(all_cpds, key=lambda t: t.variables[0])


@requires_daft
@requires_torch
def test_get_parameters2_torch(torch_models):
    est3 = torch_models["est3"]
    pseudo_counts = {
        "A": [[1], [2], [3]],
        "B": [[4], [5]],
        "C": [[6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7]],
    }
    cpds = {
        est3.estimate_cpd("A", prior_type="dirichlet", pseudo_counts=pseudo_counts["A"]),
        est3.estimate_cpd("B", prior_type="dirichlet", pseudo_counts=pseudo_counts["B"]),
        est3.estimate_cpd("C", prior_type="dirichlet", pseudo_counts=pseudo_counts["C"]),
    }
    all_cpds = est3.get_parameters(prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1)
    assert sorted(cpds, key=lambda t: t.variables[0]) == sorted(all_cpds, key=lambda t: t.variables[0])


@requires_daft
@requires_torch
def test_get_parameters3_torch(torch_models):
    est3 = torch_models["est3"]
    pseudo_counts = 0.1
    cpds = {
        est3.estimate_cpd("A", prior_type="dirichlet", pseudo_counts=pseudo_counts),
        est3.estimate_cpd("B", prior_type="dirichlet", pseudo_counts=pseudo_counts),
        est3.estimate_cpd("C", prior_type="dirichlet", pseudo_counts=pseudo_counts),
    }
    all_cpds = est3.get_parameters(prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1)

    assert sorted(cpds, key=lambda t: t.variables[0]) == sorted(all_cpds, key=lambda t: t.variables[0])
