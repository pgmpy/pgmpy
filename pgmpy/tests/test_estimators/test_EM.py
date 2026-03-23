import warnings

import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.example_models import load_model
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import compat_fns


@pytest.fixture
def cancer_models():
    model1 = load_model("bnlearn/cancer")
    data1 = model1.simulate(int(1e4), seed=42)

    model2 = DiscreteBayesianNetwork(model1.edges(), latents={"Smoker"})
    model2.add_cpds(*model1.cpds)
    data2 = model2.simulate(int(1e4), seed=42)

    yield model1, data1, model2, data2

    get_reusable_executor().shutdown(wait=True)


class TestEM:
    def test_get_parameters(self, cancer_models):
        model1, data1, model2, data2 = cancer_models

        # All observed
        est = EM(model1, data1)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

        # Latent variables
        est = EM(model2, data2)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model2.get_cpds(var)

            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

    def test_get_parameters_smoothing_k2(self, cancer_models):
        model1, data1, model2, data2 = cancer_models

        # All observed
        est = EM(model1, data1)
        cpds = est.get_parameters(
            seed=42,
            n_jobs=1,
            apply_smoothing=True,
            prior_type="k2",
            show_progress=False,
        )
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

        # Latent variables
        est = EM(model2, data2)
        cpds = est.get_parameters(
            seed=42,
            n_jobs=1,
            apply_smoothing=True,
            prior_type="k2",
            show_progress=False,
        )
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model2.get_cpds(var)

            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

    def test_get_parameters_smoothing_bdeu(self, cancer_models):
        model1, data1, model2, data2 = cancer_models

        # All observed
        est = EM(model1, data1)
        cpds = est.get_parameters(
            seed=42,
            n_jobs=1,
            apply_smoothing=True,
            prior_type="bdeu",
            equivalent_sample_size=1,
            show_progress=False,
        )
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

        # Latent variables
        est = EM(model2, data2)
        cpds = est.get_parameters(
            seed=42,
            n_jobs=1,
            apply_smoothing=True,
            prior_type="bdeu",
            equivalent_sample_size=1,
            show_progress=False,
        )
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model2.get_cpds(var)

            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

    def test_get_parameters_initial_cpds(self, cancer_models):
        model1, data1, model2, data2 = cancer_models

        # All observed. Specify initial CPDs.
        est = EM(model1, data1)
        smoker_initial = TabularCPD("Smoker", 2, [[0.1], [0.9]], state_names={"Smoker": ["True", "False"]})
        cpds = est.get_parameters(init_cpds={"Smoker": smoker_initial}, seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

        # With latents. Specify initial CPDs only for latent.
        est = EM(model2, data2)
        cpds = est.get_parameters(init_cpds={"Smoker": smoker_initial}, seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]

            # The latent variable doesn't converge to the true value when
            # the initial CPD is specified.
            if orig_cpd.variables[0] == "Smoker":
                assert np.allclose(est_cpd.values, np.array([0.123, 0.877]), atol=0.01)
            else:
                assert orig_cpd.__eq__(est_cpd, atol=0.1)

        # With latents. Specify initial CPDs for both latents and observed.
        est = EM(model2, data2)
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
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]

            # The latent variable doesn't converge to the true value when
            # the initial CPD is specified.
            if orig_cpd.variables[0] == "Smoker":
                assert np.allclose(est_cpd.values, np.array([0.123, 0.877]), atol=0.01)
            elif orig_cpd.variables[0] == "Xray":
                assert np.allclose(
                    est_cpd.values,
                    np.array([[0.799, 0.093], [0.201, 0.907]]),
                    atol=0.01,
                )
            else:
                assert orig_cpd.__eq__(est_cpd, atol=0.1)

    def test_em_init_missing_data_handling(self, cancer_models):
        model1, data1, model2, data2 = cancer_models

        df = pd.DataFrame({"A": [1, 2, 3], "B": [None, None, None], "C": [1, None, 3], "D": [4, 5, 6]})

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            est = EM(model1, df)

        # Data shape and column removal
        assert est.data.shape == (2, 3)
        assert "B" not in est.data.columns

    def test_get_parameters_random_init_cpds(self, cancer_models):
        model1, data1, model2, data2 = cancer_models

        est = EM(model1, data1)
        cpds = est.get_parameters(init_cpds="random", seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

    def test_get_parameters_uniform_init_cpds(self, cancer_models):
        model1, data1, model2, data2 = cancer_models

        est = EM(model1, data1)
        cpds = est.get_parameters(init_cpds="uniform", n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

    def test_get_parameters_node_specific_ess_bdeu(self, cancer_models):
        model1, data1, model2, data2 = cancer_models

        # All observed
        est = EM(model1, data1)
        ess_dict = {"Smoker": 10, "Cancer": 5, "Xray": 8}
        cpds = est.get_parameters(
            seed=42,
            n_jobs=1,
            apply_smoothing=True,
            prior_type="bdeu",
            equivalent_sample_size=ess_dict,
            show_progress=False,
        )
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

        # With latent variables
        est = EM(model2, data2)
        cpds = est.get_parameters(
            seed=42,
            n_jobs=1,
            apply_smoothing=True,
            prior_type="bdeu",
            equivalent_sample_size=ess_dict,
            show_progress=False,
        )
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model2.get_cpds(var)

            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

    def test_get_parameters_ess_dict_vs_scalar(self, cancer_models):
        model1, data1, model2, data2 = cancer_models

        ess_value = 7
        ess_dict = {"Smoker": ess_value, "Cancer": ess_value, "Xray": ess_value}

        est_scalar = EM(model1, data1)
        cpds_scalar = est_scalar.get_parameters(
            seed=42,
            n_jobs=1,
            apply_smoothing=True,
            prior_type="bdeu",
            equivalent_sample_size=ess_value,
            show_progress=False,
        )

        est_dict = EM(model1, data1)
        cpds_dict = est_dict.get_parameters(
            seed=42,
            n_jobs=1,
            apply_smoothing=True,
            prior_type="bdeu",
            equivalent_sample_size=ess_dict,
            show_progress=False,
        )

        # Results should be identical
        for cpd_scalar, cpd_dict in zip(
            sorted(cpds_scalar, key=lambda x: x.variables[0]),
            sorted(cpds_dict, key=lambda x: x.variables[0]),
        ):
            assert cpd_scalar.__eq__(cpd_dict, atol=1e-6)


@pytest.fixture
def cancer_models_torch():
    if not _check_soft_dependencies("torch", severity="none"):
        pytest.skip("torch not installed")

    config.set_backend("torch")

    model1 = load_model("bnlearn/cancer")
    data1 = model1.simulate(int(1e4), seed=42)

    model2 = DiscreteBayesianNetwork(model1.edges(), latents={"Smoker"})
    model2.add_cpds(*model1.cpds)
    data2 = model2.simulate(int(1e4), seed=42)

    yield model1, data1, model2, data2

    get_reusable_executor().shutdown(wait=True)
    config.set_backend("numpy")


class TestEMTorch:
    def test_get_parameters(self, cancer_models_torch):
        model1, data1, model2, data2 = cancer_models_torch

        est = EM(model1, data1)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

        est = EM(model2, data2)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model2.get_cpds(var)

            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]

            assert orig_cpd.__eq__(est_cpd, atol=0.1)

    def test_get_parameters_initial_cpds(self, cancer_models_torch):
        model1, data1, model2, data2 = cancer_models_torch

        # All observed. Specify initial CPDs.
        est = EM(model1, data1)
        smoker_initial = TabularCPD("Smoker", 2, [[0.1], [0.9]], state_names={"Smoker": ["True", "False"]})
        cpds = est.get_parameters(init_cpds={"Smoker": smoker_initial}, seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            assert orig_cpd.__eq__(est_cpd, atol=0.1)

        # With latents. Specify initial CPDs only for latent.
        est = EM(model2, data2)
        cpds = est.get_parameters(init_cpds={"Smoker": smoker_initial}, seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]

            # The latent variable doesn't converge to the true value when
            # the initial CPD is specified.
            if orig_cpd.variables[0] == "Smoker":
                assert np.allclose(
                    compat_fns.to_numpy(est_cpd.values),
                    np.array([0.123, 0.877]),
                    atol=0.01,
                )
            else:
                assert orig_cpd.__eq__(est_cpd, atol=0.1)

        # With latents. Specify initial CPDs for both latents and observed.
        est = EM(model2, data2)
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
            var = est_cpd.variables[0]
            orig_cpd = model1.get_cpds(var)
            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]

            # The latent variable doesn't converge to the true value when
            # the initial CPD is specified.
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
