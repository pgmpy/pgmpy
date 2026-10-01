import warnings

import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.example_models import load_model
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import compat_fns

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    if request.param == "torch":
        if not _check_soft_dependencies("torch", severity="none"):
            pytest.skip("torch not installed")
        config.set_backend("torch")
    yield request.param
    config.set_backend("numpy")


@pytest.fixture
def em_data():
    model1 = load_model("bnlearn/cancer")
    data1 = model1.simulate(int(1e4), seed=42)

    model2 = DiscreteBayesianNetwork(model1.edges(), latents={"Smoker"})
    model2.add_cpds(*model1.cpds)
    data2 = model2.simulate(int(1e4), seed=42)

    yield {"model1": model1, "data1": data1, "model2": model2, "data2": data2}

    get_reusable_executor().shutdown(wait=True)


def assert_cpds_close(est_cpds, model, atol=0.1, flip_smoker=False):
    for est_cpd in est_cpds:
        orig_cpd = model.get_cpds(est_cpd.variables[0])
        if flip_smoker and "Smoker" in orig_cpd.variables:
            orig_cpd.state_names["Smoker"] = [1, 0]
        assert orig_cpd.__eq__(est_cpd, atol=atol)


def test_get_parameters(em_data, backend):

    est = EM(em_data["model1"], em_data["data1"])
    cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
    assert_cpds_close(cpds, em_data["model1"])

    est = EM(em_data["model2"], em_data["data2"])
    cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
    assert_cpds_close(cpds, em_data["model2"], flip_smoker=True)


def test_get_parameters_smoothing_k2(em_data, backend):

    est = EM(em_data["model1"], em_data["data1"])
    cpds = est.get_parameters(
        seed=42,
        n_jobs=1,
        apply_smoothing=True,
        prior_type="k2",
        show_progress=False,
    )
    assert_cpds_close(cpds, em_data["model1"])

    est = EM(em_data["model2"], em_data["data2"])
    cpds = est.get_parameters(
        seed=42,
        n_jobs=1,
        apply_smoothing=True,
        prior_type="k2",
        show_progress=False,
    )
    assert_cpds_close(cpds, em_data["model2"], flip_smoker=True)


def test_get_parameters_smoothing_bdeu(em_data, backend):

    est = EM(em_data["model1"], em_data["data1"])
    cpds = est.get_parameters(
        seed=42,
        n_jobs=1,
        apply_smoothing=True,
        prior_type="bdeu",
        equivalent_sample_size=1,
        show_progress=False,
    )
    assert_cpds_close(cpds, em_data["model1"])

    est = EM(em_data["model2"], em_data["data2"])
    cpds = est.get_parameters(
        seed=42,
        n_jobs=1,
        apply_smoothing=True,
        prior_type="bdeu",
        equivalent_sample_size=1,
        show_progress=False,
    )
    assert_cpds_close(cpds, em_data["model2"], flip_smoker=True)


def test_get_parameters_initial_cpds(em_data, backend):

    est = EM(em_data["model1"], em_data["data1"])
    smoker_initial = TabularCPD("Smoker", 2, [[0.1], [0.9]], state_names={"Smoker": ["True", "False"]})
    cpds = est.get_parameters(init_cpds={"Smoker": smoker_initial}, seed=42, n_jobs=1, show_progress=False)
    assert_cpds_close(cpds, em_data["model1"])

    est = EM(em_data["model2"], em_data["data2"])
    cpds = est.get_parameters(init_cpds={"Smoker": smoker_initial}, seed=42, n_jobs=1, show_progress=False)
    for est_cpd in cpds:
        orig_cpd = em_data["model1"].get_cpds(est_cpd.variables[0])
        if "Smoker" in orig_cpd.variables:
            orig_cpd.state_names["Smoker"] = [1, 0]

        if orig_cpd.variables[0] == "Smoker":
            assert np.allclose(
                compat_fns.to_numpy(est_cpd.values),
                np.array([0.123, 0.877]),
                atol=0.01,
            )
        else:
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

    est = EM(em_data["model2"], em_data["data2"])
    xray_initial = TabularCPD(
        variable="Xray",
        variable_card=2,
        values=[[0.1, 0.8], [0.9, 0.2]],
        evidence=["Cancer"],
        evidence_card=[2],
        state_names={"Xray": ["positive", "negative"], "Cancer": ["True", "False"]},
    )
    cpds = est.get_parameters(
        init_cpds={"Smoker": smoker_initial, "Xray": xray_initial},
        seed=42,
        n_jobs=1,
        show_progress=False,
    )

    for est_cpd in cpds:
        orig_cpd = em_data["model1"].get_cpds(est_cpd.variables[0])
        if "Smoker" in orig_cpd.variables:
            orig_cpd.state_names["Smoker"] = [1, 0]

        if orig_cpd.variables[0] == "Smoker":
            assert np.allclose(
                compat_fns.to_numpy(est_cpd.values),
                np.array([0.123, 0.877]),
                atol=0.01,
            )
        elif orig_cpd.variables[0] == "Xray":
            assert np.allclose(
                compat_fns.to_numpy(est_cpd.values),
                np.array([[0.799, 0.093], [0.201, 0.907]]),
                atol=0.01,
            )
        else:
            assert orig_cpd.__eq__(est_cpd, atol=0.1)


def test_em_init_missing_data_handling(em_data, backend):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [None, None, None], "C": [1, None, 3], "D": [4, 5, 6]})

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        est = EM(em_data["model1"], df)

    assert est.data.shape == (2, 3)
    assert "B" not in est.data.columns


def test_get_parameters_random_init_cpds(em_data, backend):
    est = EM(em_data["model1"], em_data["data1"])
    cpds = est.get_parameters(init_cpds="random", seed=42, n_jobs=1, show_progress=False)
    assert_cpds_close(cpds, em_data["model1"])


def test_get_parameters_uniform_init_cpds(em_data, backend):
    est = EM(em_data["model1"], em_data["data1"])
    cpds = est.get_parameters(init_cpds="uniform", n_jobs=1, show_progress=False)
    assert_cpds_close(cpds, em_data["model1"])


def test_get_parameters_node_specific_ess_bdeu(em_data, backend):
    """Test EM with node-specific equivalent_sample_size dict for BDeu."""

    est = EM(em_data["model1"], em_data["data1"])
    ess_dict = {"Smoker": 10, "Cancer": 5, "Xray": 8}
    cpds = est.get_parameters(
        seed=42,
        n_jobs=1,
        apply_smoothing=True,
        prior_type="bdeu",
        equivalent_sample_size=ess_dict,
        show_progress=False,
    )
    assert_cpds_close(cpds, em_data["model1"])

    est = EM(em_data["model2"], em_data["data2"])
    cpds = est.get_parameters(
        seed=42,
        n_jobs=1,
        apply_smoothing=True,
        prior_type="bdeu",
        equivalent_sample_size=ess_dict,
        show_progress=False,
    )
    assert_cpds_close(cpds, em_data["model2"], flip_smoker=True)


def test_get_parameters_ess_dict_vs_scalar(em_data, backend):
    """Test that uniform ESS dict matches scalar ESS."""
    ess_value = 7
    ess_dict = dict.fromkeys(["Pollution", "Smoker", "Cancer", "Xray", "Dyspnoea"], ess_value)

    est_scalar = EM(em_data["model1"], em_data["data1"])
    cpds_scalar = est_scalar.get_parameters(
        seed=42,
        n_jobs=1,
        apply_smoothing=True,
        prior_type="bdeu",
        equivalent_sample_size=ess_value,
        show_progress=False,
    )

    est_dict = EM(em_data["model1"], em_data["data1"])
    cpds_dict = est_dict.get_parameters(
        seed=42,
        n_jobs=1,
        apply_smoothing=True,
        prior_type="bdeu",
        equivalent_sample_size=ess_dict,
        show_progress=False,
    )

    for cpd_scalar, cpd_dict in zip(
        sorted(cpds_scalar, key=lambda x: x.variables[0]),
        sorted(cpds_dict, key=lambda x: x.variables[0]),
    ):
        assert cpd_scalar.__eq__(cpd_dict, atol=1e-6)


def test_init_invalid_model_raises():
    df = pd.DataFrame({"A": [0, 1, 0], "B": [1, 1, 0]})
    with pytest.raises(NotImplementedError, match="only implemented for DAG"):
        EM("not-a-model", df)


def test_init_latent_role_assignment():
    model = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])
    df = pd.DataFrame(
        {
            "A": [0, 1, 0, 1],
            "B": [None, None, None, None],
            "C": [1, 0, 1, None],
        }
    )
    with pytest.warns(UserWarning, match="all missing values"):
        with pytest.warns(UserWarning, match="rows with missing values"):
            est = EM(model, df)
    assert "B" in est.model.latents
    assert est.data.shape == (3, 2)


def test_init_dag_conversion():
    dag = DAG([("A", "B")], latents={"B"})
    df = pd.DataFrame({"A": [0, 1, 0, 1]})
    est = EM(dag, df)
    assert isinstance(est.model, DiscreteBayesianNetwork)
    assert set(est.model.nodes()) == {"A", "B"}
    assert est.model.latents == {"B"}


def test_get_parameters_latent_card(em_data, backend):
    est = EM(em_data["model2"], em_data["data2"])
    cpds = est.get_parameters(latent_card={"Smoker": 2}, seed=42, n_jobs=1, show_progress=False)
    assert_cpds_close(cpds, em_data["model2"], flip_smoker=True)
