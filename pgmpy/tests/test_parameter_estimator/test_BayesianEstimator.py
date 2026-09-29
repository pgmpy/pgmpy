import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator


def get_cpd(estimator, variable):
    return next(cpd for cpd in estimator.parameters_ if cpd.variable == variable)


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

    est1 = DiscreteBayesianEstimator().fit(m1, d1)
    est2 = DiscreteBayesianEstimator(state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]}).fit(m1, d1)
    est3 = DiscreteBayesianEstimator().fit(m1, d2)
    return {
        "m1": m1,
        "model_latent": model_latent,
        "dag_with_latents": dag_with_latents,
        "d1": d1,
        "d2": d2,
        "est1": est1,
        "est2": est2,
        "est3": est3,
    }


def test_error_latent_model(models):
    with pytest.raises(ValueError):
        DiscreteBayesianEstimator().fit(models["model_latent"], models["d1"])
    with pytest.raises(ValueError):
        DiscreteBayesianEstimator().fit(models["dag_with_latents"], models["d1"])


def test_estimate_cpd_dirichlet(models):
    cpd_A = get_cpd(
        DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts={"A": [[0], [1]]}).fit(
            models["m1"], models["d1"]
        ),
        "A",
    )
    cpd_A_exp = TabularCPD(
        variable="A",
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={"A": [0, 1]},
    )

    assert cpd_A == cpd_A_exp

    cpd_A = get_cpd(
        DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts={"A": np.array([[0], [1]])}).fit(
            models["m1"], models["d1"]
        ),
        "A",
    )
    assert cpd_A == cpd_A_exp

    cpd_B = get_cpd(
        DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts={"B": [[9], [3]]}).fit(
            models["m1"], models["d1"]
        ),
        "B",
    )
    cpd_B_exp = TabularCPD("B", 2, [[11.0 / 15], [4.0 / 15]], state_names={"B": [0, 1]})
    assert cpd_B == cpd_B_exp

    cpd_C = get_cpd(
        DiscreteBayesianEstimator(
            prior_type="dirichlet",
            pseudo_counts={"C": [[0.4, 0.4, 0.4, 0.4], [0.6, 0.6, 0.6, 0.6]]},
        ).fit(models["m1"], models["d1"]),
        "C",
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
    cpd_C = get_cpd(
        DiscreteBayesianEstimator(
            prior_type="dirichlet",
            pseudo_counts={"C": [[0, 0, 0, 0], [0, 0, 0, 0]]},
        ).fit(models["m1"], models["d1"]),
        "C",
    )
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
    est2 = DiscreteBayesianEstimator(
        state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]},
        prior_type="BDeu",
        equivalent_sample_size=9,
    ).fit(models["m1"], models["d1"])
    est3 = DiscreteBayesianEstimator(prior_type="K2").fit(models["m1"], models["d2"])

    cpd_C1 = get_cpd(est2, "C")
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

    cpd_C2 = get_cpd(est3, "C")
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
    # Default BDeu (ess=5) on d2; columns of C ordered (A=0,B=X),(A=0,B=Y),(A=1,B=X),(A=1,B=Y),(A=2,B=X),(A=2,B=Y)
    est3 = models["est3"]
    assert len(est3.parameters_) == 3
    np.testing.assert_allclose(get_cpd(est3, "A").get_values(), [[4 / 9], [11 / 45], [14 / 45]], atol=1e-6)
    np.testing.assert_allclose(get_cpd(est3, "B").get_values(), [[0.5], [0.5]], atol=1e-6)
    np.testing.assert_allclose(
        get_cpd(est3, "C").get_values(),
        [[0.5, 29 / 46, 5 / 22, 17 / 22, 29 / 34, 17 / 22], [0.5, 17 / 46, 17 / 22, 5 / 22, 5 / 34, 5 / 22]],
        atol=1e-6,
    )


def test_get_parameters2(models):
    pseudo_counts = {
        "A": [[1], [2], [3]],
        "B": [[4], [5]],
        "C": [[6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7]],
    }
    est3 = DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts=pseudo_counts).fit(
        models["m1"], models["d2"]
    )
    assert len(est3.parameters_) == 3
    np.testing.assert_allclose(get_cpd(est3, "A").get_values(), [[3 / 8], [1 / 4], [3 / 8]], atol=1e-6)
    np.testing.assert_allclose(get_cpd(est3, "B").get_values(), [[9 / 19], [10 / 19]], atol=1e-6)
    np.testing.assert_allclose(
        get_cpd(est3, "C").get_values(),
        [[7 / 15, 0.5, 3 / 7, 0.5, 8 / 15, 0.5], [8 / 15, 0.5, 4 / 7, 0.5, 7 / 15, 0.5]],
        atol=1e-6,
    )


def test_get_parameters3(models):
    est3 = DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts=0.1).fit(models["m1"], models["d2"])
    assert len(est3.parameters_) == 3
    np.testing.assert_allclose(get_cpd(est3, "A").get_values(), [[51 / 103], [21 / 103], [31 / 103]], atol=1e-6)
    np.testing.assert_allclose(get_cpd(est3, "B").get_values(), [[0.5], [0.5]], atol=1e-6)
    np.testing.assert_allclose(
        get_cpd(est3, "C").get_values(),
        [[0.5, 21 / 32, 1 / 12, 11 / 12, 21 / 22, 11 / 12], [0.5, 11 / 32, 11 / 12, 1 / 12, 1 / 22, 1 / 12]],
        atol=1e-6,
    )


def test_node_specific_equivalent_sample_size(models):
    ess_dict = {"A": 10, "B": 20, "C": 15}
    est3 = DiscreteBayesianEstimator(prior_type="bdeu", equivalent_sample_size=ess_dict).fit(models["m1"], models["d2"])
    cpds_dict = est3.parameters_
    cpds_manual = {
        get_cpd(
            DiscreteBayesianEstimator(prior_type="bdeu", equivalent_sample_size=10).fit(models["m1"], models["d2"]),
            "A",
        ),
        get_cpd(
            DiscreteBayesianEstimator(prior_type="bdeu", equivalent_sample_size=20).fit(models["m1"], models["d2"]),
            "B",
        ),
        get_cpd(
            DiscreteBayesianEstimator(prior_type="bdeu", equivalent_sample_size=15).fit(models["m1"], models["d2"]),
            "C",
        ),
    }
    assert set(cpds_dict) == cpds_manual


def test_node_specific_ess_partial_dict(models):
    """Test that unspecified nodes default to 0 (or equivalent behavior) when dict is partial."""
    ess_dict = {"A": 10, "C": 15}
    est3 = DiscreteBayesianEstimator(prior_type="bdeu", equivalent_sample_size=ess_dict).fit(models["m1"], models["d2"])
    cpd_A_dict = get_cpd(est3, "A")
    cpd_B_dict = get_cpd(est3, "B")
    cpd_C_dict = get_cpd(est3, "C")

    cpd_A_manual = get_cpd(
        DiscreteBayesianEstimator(prior_type="bdeu", equivalent_sample_size=10).fit(models["m1"], models["d2"]),
        "A",
    )
    cpd_C_manual = get_cpd(
        DiscreteBayesianEstimator(prior_type="bdeu", equivalent_sample_size=15).fit(models["m1"], models["d2"]),
        "C",
    )
    cpd_B_manual = get_cpd(
        DiscreteBayesianEstimator(prior_type="bdeu", equivalent_sample_size=0).fit(models["m1"], models["d2"]),
        "B",
    )

    assert cpd_A_dict == cpd_A_manual
    assert cpd_B_dict == cpd_B_manual
    assert cpd_C_dict == cpd_C_manual


def test_node_specific_ess_matches_uniform_ess(models):
    """Test that uniform ESS dict matches scalar ESS."""
    ess_value = 12
    ess_dict = {"A": ess_value, "B": ess_value, "C": ess_value}
    cpds_scalar = (
        DiscreteBayesianEstimator(prior_type="bdeu", equivalent_sample_size=ess_value)
        .fit(models["m1"], models["d2"])
        .parameters_
    )
    cpds_dict = (
        DiscreteBayesianEstimator(prior_type="bdeu", equivalent_sample_size=ess_dict)
        .fit(models["m1"], models["d2"])
        .parameters_
    )
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
    est1 = DiscreteBayesianEstimator().fit(m1, d1)
    est2 = DiscreteBayesianEstimator(state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]}).fit(m1, d1)
    est3 = DiscreteBayesianEstimator().fit(m1, d2)
    yield {
        "m1": m1,
        "model_latent": model_latent,
        "d1": d1,
        "d2": d2,
        "est1": est1,
        "est2": est2,
        "est3": est3,
    }
    config.set_backend("numpy")


@requires_torch
def test_error_latent_model_torch(torch_models):
    with pytest.raises(ValueError):
        DiscreteBayesianEstimator().fit(torch_models["model_latent"], torch_models["d1"])


@requires_daft
@requires_torch
def test_estimate_cpd_dirichlet_torch(torch_models):
    cpd_A = get_cpd(
        DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts={"A": [[0], [1]]}).fit(
            torch_models["m1"], torch_models["d1"]
        ),
        "A",
    )
    cpd_A_exp = TabularCPD(
        variable="A",
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={"A": [0, 1]},
    )
    assert cpd_A == cpd_A_exp

    cpd_A = get_cpd(
        DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts={"A": np.array([[0], [1]])}).fit(
            torch_models["m1"], torch_models["d1"]
        ),
        "A",
    )
    assert cpd_A == cpd_A_exp

    cpd_B = get_cpd(
        DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts={"B": [[9], [3]]}).fit(
            torch_models["m1"], torch_models["d1"]
        ),
        "B",
    )
    cpd_B_exp = TabularCPD("B", 2, [[11.0 / 15], [4.0 / 15]], state_names={"B": [0, 1]})
    assert cpd_B == cpd_B_exp

    cpd_C = get_cpd(
        DiscreteBayesianEstimator(
            prior_type="dirichlet",
            pseudo_counts={"C": [[0.4, 0.4, 0.4, 0.4], [0.6, 0.6, 0.6, 0.6]]},
        ).fit(torch_models["m1"], torch_models["d1"]),
        "C",
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
    cpd_C = get_cpd(
        DiscreteBayesianEstimator(
            prior_type="dirichlet",
            pseudo_counts={"C": [[0, 0, 0, 0], [0, 0, 0, 0]]},
        ).fit(torch_models["m1"], torch_models["d1"]),
        "C",
    )
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
    est2 = DiscreteBayesianEstimator(
        state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]},
        prior_type="BDeu",
        equivalent_sample_size=9,
    ).fit(torch_models["m1"], torch_models["d1"])
    est3 = DiscreteBayesianEstimator(prior_type="K2").fit(torch_models["m1"], torch_models["d2"])
    cpd_C1 = get_cpd(est2, "C")
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

    cpd_C2 = get_cpd(est3, "C")
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
    cpds = [get_cpd(est3, "A"), get_cpd(est3, "B"), get_cpd(est3, "C")]
    all_cpds = est3.parameters_
    assert sorted(cpds, key=lambda t: t.variables[0]) == sorted(all_cpds, key=lambda t: t.variables[0])


@requires_daft
@requires_torch
def test_get_parameters2_torch(torch_models):
    pseudo_counts = {
        "A": [[1], [2], [3]],
        "B": [[4], [5]],
        "C": [[6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7]],
    }
    est3 = DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts=pseudo_counts).fit(
        torch_models["m1"], torch_models["d2"]
    )
    cpds = {get_cpd(est3, "A"), get_cpd(est3, "B"), get_cpd(est3, "C")}
    all_cpds = est3.parameters_
    assert sorted(cpds, key=lambda t: t.variables[0]) == sorted(all_cpds, key=lambda t: t.variables[0])


@requires_daft
@requires_torch
def test_get_parameters3_torch(torch_models):
    est3 = DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts=0.1).fit(
        torch_models["m1"], torch_models["d2"]
    )
    cpds = {get_cpd(est3, "A"), get_cpd(est3, "B"), get_cpd(est3, "C")}
    all_cpds = est3.parameters_

    assert sorted(cpds, key=lambda t: t.variables[0]) == sorted(all_cpds, key=lambda t: t.variables[0])
