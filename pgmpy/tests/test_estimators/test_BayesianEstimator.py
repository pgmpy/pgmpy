import pytest

import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


@pytest.fixture
def model_setup():
    m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
    model_latent = DiscreteBayesianNetwork(
        [("A", "C"), ("B", "C")], latents=["C"]
    )
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
    est2 = BayesianEstimator(
        m1, d1, state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]}
    )
    est3 = BayesianEstimator(m1, d2)

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


@pytest.fixture(autouse=True)
def cleanup():
    yield
    get_reusable_executor().shutdown(wait=True)


class TestBayesianEstimator:
    def test_error_latent_model(self, model_setup):
        with pytest.raises(ValueError):
            BayesianEstimator(model_setup["model_latent"], model_setup["d1"])
        with pytest.raises(ValueError):
            BayesianEstimator(model_setup["dag_with_latents"], model_setup["d1"])

    def test_estimate_cpd_dirichlet(self, model_setup):
        cpd_A = model_setup["est1"].estimate_cpd(
            "A", prior_type="dirichlet", pseudo_counts=[[0], [1]]
        )
        cpd_A_exp = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"A": [0, 1]},
        )
        assert cpd_A == cpd_A_exp

        # also test passing pseudo_counts as np.array
        pseudo_counts = np.array([[0], [1]])
        cpd_A = model_setup["est1"].estimate_cpd(
            "A", prior_type="dirichlet", pseudo_counts=pseudo_counts
        )
        assert cpd_A == cpd_A_exp

        cpd_B = model_setup["est1"].estimate_cpd(
            "B", prior_type="dirichlet", pseudo_counts=[[9], [3]]
        )
        cpd_B_exp = TabularCPD(
            "B", 2, [[11.0 / 15], [4.0 / 15]], state_names={"B": [0, 1]}
        )
        assert cpd_B == cpd_B_exp

        cpd_C = model_setup["est1"].estimate_cpd(
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

    def test_estimate_cpd_improper_prior(self, model_setup):
        cpd_C = model_setup["est1"].estimate_cpd(
            "C", prior_type="dirichlet", pseudo_counts=[[0, 0, 0, 0], [0, 0, 0, 0]]
        )
        cpd_C_correct = TabularCPD(
            "C",
            2,
            [[0.0, 0.0, 1.0, np.nan], [1.0, 1.0, 0.0, np.nan]],
            evidence=["A", "B"],
            evidence_card=[2, 2],
            state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
        )
        # manual comparison because np.nan != np.nan
        assert (
            (cpd_C.values == cpd_C_correct.values)
            | np.isnan(cpd_C.values) & np.isnan(cpd_C_correct.values)
        ).all()

    def test_estimate_cpd_shortcuts(self, model_setup):
        cpd_C1 = model_setup["est2"].estimate_cpd(
            "C", prior_type="BDeu", equivalent_sample_size=9
        )
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

        cpd_C2 = model_setup["est3"].estimate_cpd("C", prior_type="K2")
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

    def test_get_parameters(self, model_setup):
        cpds = set(
            [
                model_setup["est3"].estimate_cpd("A"),
                model_setup["est3"].estimate_cpd("B"),
                model_setup["est3"].estimate_cpd("C"),
            ]
        )
        assert set(model_setup["est3"].get_parameters(n_jobs=1)) == cpds

    def test_get_parameters2(self, model_setup):
        pseudo_counts = {
            "A": [[1], [2], [3]],
            "B": [[4], [5]],
            "C": [[6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7]],
        }
        cpds = set(
            [
                model_setup["est3"].estimate_cpd(
                    "A", prior_type="dirichlet", pseudo_counts=pseudo_counts["A"]
                ),
                model_setup["est3"].estimate_cpd(
                    "B", prior_type="dirichlet", pseudo_counts=pseudo_counts["B"]
                ),
                model_setup["est3"].estimate_cpd(
                    "C", prior_type="dirichlet", pseudo_counts=pseudo_counts["C"]
                ),
            ]
        )
        assert set(
            model_setup["est3"].get_parameters(
                prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1
            )
        ) == cpds

    def test_get_parameters3(self, model_setup):
        pseudo_counts = 0.1
        cpds = set(
            [
                model_setup["est3"].estimate_cpd(
                    "A", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
                model_setup["est3"].estimate_cpd(
                    "B", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
                model_setup["est3"].estimate_cpd(
                    "C", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
            ]
        )
        assert set(
            model_setup["est3"].get_parameters(
                prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1
            )
        ) == cpds

    def test_node_specific_equivalent_sample_size(self, model_setup):
        """Test BDeu with node-specific equivalent_sample_size dict."""
        ess_dict = {"A": 10, "B": 20, "C": 15}
        cpds_dict = model_setup["est3"].get_parameters(
            prior_type="bdeu", equivalent_sample_size=ess_dict, n_jobs=1
        )

        cpds_manual = set(
            [
                model_setup["est3"].estimate_cpd(
                    "A", prior_type="bdeu", equivalent_sample_size=10
                ),
                model_setup["est3"].estimate_cpd(
                    "B", prior_type="bdeu", equivalent_sample_size=20
                ),
                model_setup["est3"].estimate_cpd(
                    "C", prior_type="bdeu", equivalent_sample_size=15
                ),
            ]
        )
        assert set(cpds_dict) == cpds_manual

    def test_node_specific_ess_partial_dict(self, model_setup):
        """Test that unspecified nodes default to 0 (or equivalent behavior) when dict is partial."""
        ess_dict = {"A": 10, "C": 15}
        cpd_A_dict = model_setup["est3"].estimate_cpd(
            "A", prior_type="bdeu", equivalent_sample_size=ess_dict
        )
        cpd_B_dict = model_setup["est3"].estimate_cpd(
            "B", prior_type="bdeu", equivalent_sample_size=ess_dict
        )
        cpd_C_dict = model_setup["est3"].estimate_cpd(
            "C", prior_type="bdeu", equivalent_sample_size=ess_dict
        )

        cpd_A_manual = model_setup["est3"].estimate_cpd(
            "A", prior_type="bdeu", equivalent_sample_size=10
        )
        cpd_C_manual = model_setup["est3"].estimate_cpd(
            "C", prior_type="bdeu", equivalent_sample_size=15
        )
        cpd_B_manual = model_setup["est3"].estimate_cpd(
            "B", prior_type="bdeu", equivalent_sample_size=0
        )

        assert cpd_A_dict == cpd_A_manual
        assert cpd_B_dict == cpd_B_manual
        assert cpd_C_dict == cpd_C_manual

    def test_node_specific_ess_matches_uniform_ess(self, model_setup):
        """Test that uniform ESS dict matches scalar ESS."""
        ess_value = 12
        ess_dict = {"A": ess_value, "B": ess_value, "C": ess_value}
        cpds_scalar = model_setup["est3"].get_parameters(
            prior_type="bdeu", equivalent_sample_size=ess_value, n_jobs=1
        )
        cpds_dict = model_setup["est3"].get_parameters(
            prior_type="bdeu", equivalent_sample_size=ess_dict, n_jobs=1
        )
        assert set(cpds_scalar) == set(cpds_dict)


@pytest.mark.skipif(
    not _check_soft_dependencies("daft-pgm", severity="none"),
    reason="execute only if required dependency present",
)
class TestBayesianEstimatorTorch:
    @pytest.fixture(autouse=True)
    def setup_torch(self):
        config.set_backend("torch")
        m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
        model_latent = DiscreteBayesianNetwork(
            [("A", "C"), ("B", "C")], latents=["C"]
        )
        d1 = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
        d2 = pd.DataFrame(
            data={
                "A": [0, 0, 1, 0, 2, 0, 2, 1, 0, 2],
                "B": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
                "C": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            }
        )
        est1 = BayesianEstimator(m1, d1)
        est2 = BayesianEstimator(
            m1, d1, state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]}
        )
        est3 = BayesianEstimator(m1, d2)

        self.m1 = m1
        self.model_latent = model_latent
        self.d1 = d1
        self.d2 = d2
        self.est1 = est1
        self.est2 = est2
        self.est3 = est3

        yield

        del self.m1
        del self.d1
        del self.d2
        del self.est1
        del self.est2
        get_reusable_executor().shutdown(wait=True)
        config.set_backend("numpy")

    def test_error_latent_model(self):
        with pytest.raises(ValueError):
            BayesianEstimator(self.model_latent, self.d1)

    def test_estimate_cpd_dirichlet(self):
        cpd_A = self.est1.estimate_cpd(
            "A", prior_type="dirichlet", pseudo_counts=[[0], [1]]
        )
        cpd_A_exp = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"A": [0, 1]},
        )
        assert cpd_A == cpd_A_exp

        # also test passing pseudo_counts as np.array
        pseudo_counts = np.array([[0], [1]])
        cpd_A = self.est1.estimate_cpd(
            "A", prior_type="dirichlet", pseudo_counts=pseudo_counts
        )
        assert cpd_A == cpd_A_exp

        cpd_B = self.est1.estimate_cpd(
            "B", prior_type="dirichlet", pseudo_counts=[[9], [3]]
        )
        cpd_B_exp = TabularCPD(
            "B", 2, [[11.0 / 15], [4.0 / 15]], state_names={"B": [0, 1]}
        )
        assert cpd_B == cpd_B_exp

        cpd_C = self.est1.estimate_cpd(
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

    def test_estimate_cpd_improper_prior(self):
        cpd_C = self.est1.estimate_cpd(
            "C", prior_type="dirichlet", pseudo_counts=[[0, 0, 0, 0], [0, 0, 0, 0]]
        )
        cpd_C_correct = TabularCPD(
            "C",
            2,
            [[0.0, 0.0, 1.0, np.nan], [1.0, 1.0, 0.0, np.nan]],
            evidence=["A", "B"],
            evidence_card=[2, 2],
            state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
        )
        # manual comparison because np.nan != np.nan
        assert (
            (cpd_C.values == cpd_C_correct.values)
            | config.get_compute_backend().isnan(cpd_C.values)
            & config.get_compute_backend().isnan(cpd_C_correct.values)
        ).all()

    def test_estimate_cpd_shortcuts(self):
        cpd_C1 = self.est2.estimate_cpd(
            "C", prior_type="BDeu", equivalent_sample_size=9
        )
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

        cpd_C2 = self.est3.estimate_cpd("C", prior_type="K2")
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

    def test_get_parameters(self):
        cpds = [
            self.est3.estimate_cpd("A"),
            self.est3.estimate_cpd("B"),
            self.est3.estimate_cpd("C"),
        ]
        all_cpds = self.est3.get_parameters(n_jobs=1)
        assert sorted(cpds, key=lambda t: t.variables[0]) == sorted(
            all_cpds, key=lambda t: t.variables[0]
        )

    def test_get_parameters2(self):
        pseudo_counts = {
            "A": [[1], [2], [3]],
            "B": [[4], [5]],
            "C": [[6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7]],
        }
        cpds = set(
            [
                self.est3.estimate_cpd(
                    "A", prior_type="dirichlet", pseudo_counts=pseudo_counts["A"]
                ),
                self.est3.estimate_cpd(
                    "B", prior_type="dirichlet", pseudo_counts=pseudo_counts["B"]
                ),
                self.est3.estimate_cpd(
                    "C", prior_type="dirichlet", pseudo_counts=pseudo_counts["C"]
                ),
            ]
        )
        all_cpds = self.est3.get_parameters(
            prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1
        )
        assert sorted(cpds, key=lambda t: t.variables[0]) == sorted(
            all_cpds, key=lambda t: t.variables[0]
        )

    def test_get_parameters3(self):
        pseudo_counts = 0.1
        cpds = set(
            [
                self.est3.estimate_cpd(
                    "A", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
                self.est3.estimate_cpd(
                    "B", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
                self.est3.estimate_cpd(
                    "C", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
            ]
        )
        all_cpds = self.est3.get_parameters(
            prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1
        )
        assert sorted(cpds, key=lambda t: t.variables[0]) == sorted(
            all_cpds, key=lambda t: t.variables[0]
        )
