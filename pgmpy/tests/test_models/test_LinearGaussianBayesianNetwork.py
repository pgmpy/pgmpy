import io
import os

import numpy as np
import numpy.testing as np_test
import pandas as pd
import pytest

from pgmpy.example_models import load_model
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestLGBNMethods:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        self.cpd1 = LinearGaussianCPD(variable="x1", beta=[1], std=4)
        self.cpd2 = LinearGaussianCPD(variable="x2", beta=[-5, 0.5], std=4, evidence=["x1"])
        self.cpd3 = LinearGaussianCPD(variable="x3", beta=[4, -1], std=3, evidence=["x2"])

    def test_cpds_simple(self):
        assert "x1" == self.cpd1.variable
        assert 4 == self.cpd1.std
        assert [1] == self.cpd1.beta

        self.model.add_cpds(self.cpd1)
        cpd = self.model.get_cpds("x1")
        assert cpd.variable == self.cpd1.variable
        assert cpd.std == self.cpd1.std
        assert cpd.beta == self.cpd1.beta

    def test_add_cpds(self):
        self.model.add_cpds(self.cpd1)
        cpd = self.model.get_cpds("x1")
        assert cpd.variable == self.cpd1.variable
        assert cpd.std == self.cpd1.std

        self.model.add_cpds(self.cpd2)
        cpd = self.model.get_cpds("x2")
        assert cpd.variable == self.cpd2.variable
        assert cpd.std == self.cpd2.std
        assert cpd.evidence == self.cpd2.evidence

        self.model.add_cpds(self.cpd3)
        cpd = self.model.get_cpds("x3")
        assert cpd.variable == self.cpd3.variable
        assert cpd.std == self.cpd3.std
        assert cpd.evidence == self.cpd3.evidence

        tab_cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
        )
        with pytest.raises(ValueError):
            self.model.add_cpds(tab_cpd)
        with pytest.raises(ValueError):
            self.model.add_cpds(1)
        with pytest.raises(ValueError):
            self.model.add_cpds(1, tab_cpd)

    def test_remove_cpds(self):
        model = LinearGaussianBayesianNetwork([("X1", "X2"), ("X2", "X3")])

        cpd_X1 = LinearGaussianCPD("X1", beta=[1.0], std=2.0)
        cpd_X2 = LinearGaussianCPD("X2", beta=[-1.0, 0.5], std=1.5, evidence=["X1"])
        cpd_X3 = LinearGaussianCPD("X3", beta=[0.0, 1.0], std=1.0, evidence=["X2"])

        model.add_cpds(cpd_X1, cpd_X2, cpd_X3)

        assert len(model.get_cpds()) == 3

        model.remove_cpds(cpd_X2, cpd_X3)

        remaining_cpds = model.get_cpds()
        assert len(remaining_cpds) == 1
        assert remaining_cpds[0] == cpd_X1

        assert cpd_X2 not in remaining_cpds
        assert cpd_X3 not in remaining_cpds

    def test_copy(self):
        model = LinearGaussianBayesianNetwork([("A", "B"), ("B", "C")])
        cpd_a = LinearGaussianCPD(variable="A", beta=[1], std=4)
        cpd_b = LinearGaussianCPD(variable="B", beta=[-5, 0.5], std=4, evidence=["A"])
        cpd_c = LinearGaussianCPD(variable="C", beta=[4, -1], std=3, evidence=["B"])

        model.add_cpds(cpd_a, cpd_b, cpd_c)

        copy_model = model.copy()

        assert set(copy_model.nodes()) == set(model.nodes())
        assert set(copy_model.edges()) == set(model.edges())

        original_cpds = model.get_cpds()
        copied_cpds = copy_model.get_cpds()

        assert len(original_cpds) == len(copied_cpds)
        for orig_cpd, copied_cpd in zip(original_cpds, copied_cpds):
            assert orig_cpd.variable == copied_cpd.variable
            assert orig_cpd.evidence == copied_cpd.evidence
            np_test.assert_array_almost_equal(orig_cpd.beta, copied_cpd.beta)
            assert orig_cpd.std == copied_cpd.std

        model.remove_cpds(cpd_b)
        assert len(model.get_cpds()) == 2
        assert len(copy_model.get_cpds()) == 3

    def test_to_joint_gaussian(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        mean, cov = self.model.to_joint_gaussian()
        np_test.assert_array_almost_equal(mean, np.array([1.0, -4.5, 8.5]), decimal=3)
        np_test.assert_array_almost_equal(
            cov,
            np.array([[16.0, 8.0, -8.0], [8.0, 20.0, -20.0], [-8.0, -20.0, 29.0]]),
            decimal=3,
        )

    def test_check_model(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        assert self.model.check_model() == True

        self.model.add_edge("x1", "x4")
        cpd4 = LinearGaussianCPD("x4", [4, -1], 3, ["x2"])
        self.model.add_cpds(cpd4)

        with pytest.raises(ValueError):
            self.model.check_model()

    def test_not_implemented_methods(self):
        with pytest.raises(ValueError):
            self.model.get_cardinality("x1")
        with pytest.raises(NotImplementedError):
            self.model.to_markov_model()
        with pytest.raises(NotImplementedError):
            self.model.is_imap([[1, 2, 3], [1, 5, 6]])

    def test_simulate(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        df_cont = self.model.simulate(n_samples=1000000, rng=42)

        # Same model in terms of equations
        rng = np.random.default_rng(seed=42)
        x1 = 1 + rng.normal(0, 4, 1000000)
        x2 = -5 + 0.5 * x1 + rng.normal(0, 4, 1000000)
        x3 = 4 + -1 * x2 + rng.normal(0, 3, 1000000)
        df_equ = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        np_test.assert_array_almost_equal(df_cont.mean(), df_equ.mean(), decimal=1)
        np_test.assert_array_almost_equal(df_cont.cov(), df_equ.cov(), decimal=1)

    def test_simulate_with_evidence(self):

        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        evidence = {"x1": 0}
        df = self.model.simulate(n_samples=10000, rng=42, evidence=evidence)

        missing_vars, mean_cond, cov_cond = self.model.predict_probability(pd.DataFrame([evidence]))
        sorted_indices = np.argsort(missing_vars)
        missing_vars = [missing_vars[i] for i in sorted_indices]
        mean_cond = mean_cond[:, sorted_indices]
        cov_cond = cov_cond[sorted_indices][:, sorted_indices]

        rng = np.random.default_rng(seed=42)
        samples = rng.multivariate_normal(mean=mean_cond[0], cov=cov_cond, size=10000, method="cholesky")
        df_equ = pd.DataFrame(samples, columns=missing_vars)

        np_test.assert_array_almost_equal(df.mean()[["x2", "x3"]], df_equ.mean(), decimal=5)
        np_test.assert_array_almost_equal(df.cov()[["x2", "x3"]].loc[["x2", "x3"]], df_equ.cov(), decimal=5)

    def test_simulate_with_intervention(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        do = {"x2": 1.0}
        df = self.model.simulate(n_samples=1000_000, rng=42, do=do)

        rng = np.random.default_rng(seed=42)
        x1 = 1 + rng.normal(0, 4, 1000_000)
        x2 = np.full(1000_000, 1.0)
        x3 = 4 + -1 * x2 + rng.normal(0, 3, 1000_000)
        df_equ = pd.DataFrame({"x1": x1, "x3": x3, "x2": do["x2"]})

        np_test.assert_array_almost_equal(df.mean(), df_equ.mean(), decimal=1)
        np_test.assert_array_almost_equal(df.cov(), df_equ.cov(), decimal=1)

    def test_simulate_with_virtual_intervention(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)

        new_cpd = LinearGaussianCPD(variable="x2", beta=[1.0], std=2.0)
        virtual_intervention = [new_cpd]

        df = self.model.simulate(n_samples=100_000, rng=42, virtual_intervention=virtual_intervention)

        rng = np.random.default_rng(seed=42)
        x1 = 1 + rng.normal(0, 4, 100_000)
        x2 = 1 + rng.normal(0, 2, 100_000)
        x3 = 4 + -1 * x2 + rng.normal(0, 3, 100_000)

        df_equiv = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        np_test.assert_array_almost_equal(df.mean(), df_equiv.mean(), decimal=1)
        np_test.assert_array_almost_equal(df.cov(), df_equiv.cov(), decimal=1)

    def test_simulate_latents(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        self.model.latents.add("x2")

        df_with_latents = self.model.simulate(n_samples=1000, rng=42, include_latents=True)
        df_without_latents = self.model.simulate(n_samples=1000, rng=42, include_latents=False)

        for latent_var in self.model.latents:
            assert latent_var in df_with_latents.columns

        for latent_var in self.model.latents:
            assert latent_var not in df_without_latents.columns

        non_latent_vars = [node for node in self.model.nodes() if node not in self.model.latents]
        for var in non_latent_vars:
            assert var in df_with_latents.columns
            assert var in df_without_latents.columns

    def test_simulate_against_manual_results(self):
        model = LinearGaussianBayesianNetwork([("X1", "X2"), ("X1", "X3"), ("X2", "X3")])

        cpd_X1 = LinearGaussianCPD("X1", beta=[0.0], std=1.0, evidence=[])
        cpd_X2 = LinearGaussianCPD("X2", beta=[0.0, 2.0], std=1.0, evidence=["X1"])
        cpd_X3 = LinearGaussianCPD("X3", beta=[0.0, -1.0, 0.5], std=1.0, evidence=["X1", "X2"])

        model.add_cpds(cpd_X1, cpd_X2, cpd_X3)

        df = model.simulate(n_samples=100000, rng=42, do={"X1": 1.0})

        E_X1 = 1.0
        E_X2 = 2.0 * E_X1
        E_X3 = -E_X1 + 0.5 * E_X2

        expected_mean = np.array([E_X1, E_X2, E_X3])

        expected_cov = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.25]])

        sim_mean = df[["X1", "X2", "X3"]].mean().values
        sim_cov = df[["X1", "X2", "X3"]].cov().values

        np_test.assert_array_almost_equal(sim_mean, expected_mean, decimal=1)
        np_test.assert_array_almost_equal(sim_cov, expected_cov, decimal=1)

    def test_simulate_matches_ancestral_sampling(self):
        model = LinearGaussianBayesianNetwork([(z, v) for z in ("Z1", "Z2", "Z3") for v in ("X", "Y")])
        model.add_cpds(
            LinearGaussianCPD("Z1", beta=[0], std=1),
            LinearGaussianCPD("Z2", beta=[0], std=1),
            LinearGaussianCPD("Z3", beta=[0], std=1),
            LinearGaussianCPD("X", beta=[1, 0.5, 0.5, 0.5], std=1, evidence=["Z1", "Z2", "Z3"]),
            LinearGaussianCPD("Y", beta=[-1, 0.5, 0.5, 0.5], std=1, evidence=["Z1", "Z2", "Z3"]),
        )
        df = model.simulate(n_samples=100, rng=42)

        noise = pd.DataFrame(np.random.default_rng(42).standard_normal((100, 5)), columns=df.columns)
        expected = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        for var in df.columns:
            cpd = model.get_cpds(var)
            parents = expected[cpd.evidence].to_numpy()
            expected[var] = cpd.beta[0] + parents @ np.asarray(cpd.beta[1:]) + cpd.std * noise[var]

        np_test.assert_allclose(df.to_numpy(), expected.to_numpy())

    def test_simulate_raises_for_invalid_do_and_evidence_nodes(self):
        model = LinearGaussianBayesianNetwork([("A", "B")])
        cpd_a = LinearGaussianCPD("A", beta=[1], std=1.0)
        cpd_b = LinearGaussianCPD("B", beta=[0.5, 2.0], std=1.0, evidence=["A"])
        model.add_cpds(cpd_a, cpd_b)

        with pytest.raises(ValueError, match="do-nodes.*not present.*X"):
            model.simulate(n_samples=100, do={"X": 5.0})

        with pytest.raises(ValueError, match="evidence-nodes.*not present.*Y"):
            model.simulate(n_samples=100, evidence={"Y": 1.0})

        with pytest.raises(ValueError, match="can't be in both do and evidence.*A"):
            model.simulate(n_samples=100, do={"A": 1.0}, evidence={"A": 2.0})

    def test_fit(self):
        # Test fit on a simple model
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        df = self.model.simulate(n_samples=int(1e5), rng=42)
        new_model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        # breakpoint()
        new_model.fit(
            df,
        )

        for node in self.model.nodes():
            cpd_orig = self.model.get_cpds(node)
            cpd_est = new_model.get_cpds(node)

            assert cpd_orig.variable == cpd_est.variable
            assert round(cpd_orig.std, 1) == round(cpd_est.std, 1)
            assert round(cpd_orig.beta[0], 1) == round(cpd_est.beta[0], 1)

            for index, evid_var in enumerate(cpd_orig.evidence):
                est_index = cpd_est.evidence.index(evid_var)
                assert round(cpd_orig.beta[index + 1], 1) == round(cpd_est.beta[est_index + 1], 1)

        # Test fit on the alarm model
        model = load_model("bnlearn/alarm")
        model_lin = LinearGaussianBayesianNetwork(model.edges())
        cpds = model_lin.get_random_cpds(rng=42)
        model_lin.add_cpds(*cpds)
        df = model_lin.simulate(n_samples=int(1e6), rng=42)

        new_model_lin = LinearGaussianBayesianNetwork(model.edges())
        new_model_lin.fit(df)

        for node in model_lin.nodes():
            cpd_orig = model_lin.get_cpds(node)
            cpd_est = new_model_lin.get_cpds(node)

            assert cpd_orig.variable == cpd_est.variable
            assert abs(cpd_orig.std - cpd_est.std) < 0.1
            assert abs(cpd_orig.beta[0] - cpd_est.beta[0]) < 0.1

            for index, evid_var in enumerate(cpd_orig.evidence):
                est_index = cpd_est.evidence.index(evid_var)
                assert abs(cpd_orig.beta[index + 1] - cpd_est.beta[est_index + 1]) < 0.2

    def test_fit_drops_rows_with_missing_values(self):
        rng = np.random.default_rng(seed=0)
        df = pd.DataFrame({"x1": rng.normal(size=100), "x3": rng.normal(size=100)})
        df["x2"] = 2 * df["x1"] + rng.normal(size=100)
        df.loc[[3, 7], "x1"] = np.nan
        df.loc[11, "x3"] = np.nan

        model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        with self.assertWarnsRegex(UserWarning, "3 rows with missing values were dropped"):
            model.fit(df)

        expected = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        expected.fit(df.dropna())
        for node in model.nodes():
            np_test.assert_allclose(model.get_cpds(node).beta, expected.get_cpds(node).beta, rtol=1e-12)
            np_test.assert_allclose(model.get_cpds(node).std, expected.get_cpds(node).std, rtol=1e-12)

    def test_fit_invalid_estimator(self):
        new_model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        df = pd.DataFrame(np.random.randn(100, 3), columns=["x1", "x2", "x3"])
        with pytest.raises(TypeError):
            new_model.fit(df, estimator="unbiased")

    def test_predict_simple(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        df = self.model.simulate(n_samples=10, rng=42)
        df = df.drop("x2", axis=1)

        result = self.model.predict(df)

        assert isinstance(result, pd.DataFrame)

        assert list(result.columns) == ["x2"]
        assert result.shape == (10, 1)
        assert np.allclose(
            result["x2"].round(2).values,
            [-7.99, -5.11, -5.02, -5.45, -2.38, -3.43, -2.52, -2.44, -7.28, -3.54],
        )

    def test_predict_ecoli(self):
        model = load_model("bnlearn/ecoli70")
        df = model.simulate(n_samples=10, rng=18)
        df = df.drop(["yceP", "yheI", "cspA"], axis=1)
        result = model.predict(df)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"yceP", "yheI", "cspA"}
        assert result.shape == (10, 3)

        # calculated by saving df to csv and using R to predict
        # model is loaded from bnlearn, impute function from bnlearn to generate true values

        true_data = {
            "yceP": [
                2.1082,
                1.0409,
                2.7293,
                0.364,
                -1.0284,
                1.0486,
                2.2428,
                1.5194,
                1.7567,
                0.7267,
            ],
            "yheI": [
                1.3727,
                -0.5331,
                -0.4184,
                2.0929,
                2.7519,
                -1.1885,
                1.5676,
                -0.6507,
                1.8537,
                0.9404,
            ],
            "cspA": [
                2.1082,
                -0.7373,
                -0.2022,
                0.5873,
                -1.6884,
                2.1621,
                -1.0443,
                1.6545,
                0.5553,
                1.572,
            ],
        }
        for var_name, expected in true_data.items():
            assert np.allclose(
                result[var_name].round(1).values,
                np.array(expected).round(1),
            )

    def test_predict_probability_simple(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        df = self.model.simulate(n_samples=10, rng=42)
        df = df.drop("x2", axis=1)
        variables, mu, cov = self.model.predict_probability(df)

        # variables
        assert variables == ["x2"]

        # mu shape and values (same as the old test_predict_simple)
        assert mu.shape == (10, 1)
        assert np.allclose(
            mu.round(2).squeeze(),
            [-7.99, -5.11, -5.02, -5.45, -2.38, -3.43, -2.52, -2.44, -7.28, -3.54],
        )

        # cov shape and value (carried over from old test_predict_simple)
        assert cov.shape == (1, 1)
        assert cov.round(2).squeeze() == 5.76

    def test_get_random_cpds(self):
        model = load_model("bnlearn/alarm")
        model_lin = LinearGaussianBayesianNetwork(model.edges())
        cpds = model_lin.get_random_cpds()
        assert len(cpds) == len(model.nodes())
        cpds1 = model_lin.get_random_cpds(rng=None)
        cpds2 = model_lin.get_random_cpds(rng=None)
        betas1 = [cpd.beta[0] for cpd in cpds1]
        betas2 = [cpd.beta[0] for cpd in cpds2]
        assert not (betas1 == betas2)

    def test_get_random(self):
        model1 = LinearGaussianBayesianNetwork.get_random(n_nodes=10, edge_prob=0.8)
        model2 = LinearGaussianBayesianNetwork.get_random(n_nodes=10, edge_prob=0.1)
        assert model1.edges() != model2.edges()
        assert isinstance(model1, LinearGaussianBayesianNetwork), "Incorrect instance"
        assert isinstance(model2, LinearGaussianBayesianNetwork), "Incorrect instance"

        model_fixed = LinearGaussianBayesianNetwork.get_random(n_nodes=7, n_edges=6)
        assert len(model_fixed.edges()) == 6
        assert isinstance(model_fixed, LinearGaussianBayesianNetwork), "Incorrect instance"

        node_names = ["a", "aa", "aaa", "aaaa", "aaaaa"]
        model3 = LinearGaussianBayesianNetwork.get_random(n_nodes=5, edge_prob=0.5, node_names=node_names)
        assert len(model3.nodes()) == 5
        assert sorted(model3.nodes()) == node_names
        assert len(model3.cpds) == 5
        assert isinstance(model3, LinearGaussianBayesianNetwork), "Incorrect instance"

    def test_log_likelihood(self):
        # Test log likelihood with dataframes
        self.model.add_cpds(self.cpd1)
        self.model.add_cpds(self.cpd2)
        self.model.add_cpds(self.cpd3)
        np.random.seed(42)
        x1 = np.random.normal(0, 4, 100)
        x2 = -5 + 0.5 * x1 + np.random.normal(0, 4, 100)
        x3 = 4 - 1 * x2 + np.random.normal(0, 3, 100)
        good_fit = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        bad_fit = pd.DataFrame(np.random.normal(0, 10, (100, 3)), columns=["x1", "x2", "x3"])
        ll_good = self.model.log_likelihood(good_fit)
        ll_bad = self.model.log_likelihood(bad_fit)
        assert isinstance(ll_good, float), "Expected log_likelihood to return a float."
        assert isinstance(ll_bad, float), "Expected log_likelihood to return a float."
        assert ll_good > ll_bad, "Expected higher log-likelihood for well-fitting data compared to random data."

        # Wrong column names should raise error
        data_wrong_names = pd.DataFrame(
            {
                "a": np.random.normal(0, 1, 100),
                "b": np.random.normal(0, 1, 100),
                "c": np.random.normal(0, 1, 100),
            }
        )
        with pytest.raises(ValueError):
            self.model.log_likelihood(data_wrong_names)

    def test_structure_mismatch_with_same_cpds(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        other = LinearGaussianBayesianNetwork([("x1", "x3"), ("x3", "x2")])
        other.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        assert self.model != other


class TestLGBNCreation:
    def test_class_init_with_adj_matrix_dict_of_dict(self):
        adj = {"a": {"b": 4, "c": 3}, "b": {"c": 2}}
        self.graph = LinearGaussianBayesianNetwork(adj, latents={"a"})
        assert self.graph.latents == set("a")
        assert sorted(self.graph.nodes()) == ["a", "b", "c"]
        assert self.graph.adj["a"]["c"]["weight"] == 3

    def test_class_init_with_adj_matrix_dict_of_list(self):
        adj = {"a": ["b", "c"], "b": ["c"]}
        self.graph = LinearGaussianBayesianNetwork(adj, latents={"a"})
        assert self.graph.latents == set("a")
        assert sorted(self.graph.nodes()) == ["a", "b", "c"]

    def test_class_init_with_pd_adj_df(self):
        df = pd.DataFrame([[0, 3], [0, 0]])
        self.graph = LinearGaussianBayesianNetwork(df, latents={0})
        assert self.graph.latents == {0}
        assert sorted(self.graph.nodes()) == [0, 1]
        assert self.graph.adj[0][1]["weight"] == {"weight": 3}


class TestDAGParser:
    def test_from_lavaan(self):
        model_str = "d ~ i"
        model_from_str = LinearGaussianBayesianNetwork.from_lavaan(string=model_str)
        expected_edges = {("i", "d")}
        assert set(model_from_str.edges()) == expected_edges

    def test_from_dagitty(self):
        model_str = """dag{ smoking -> "carry matches" }"""
        model_from_str = LinearGaussianBayesianNetwork.from_dagitty(string=model_str)
        expected_edges = {("smoking", "carry matches")}
        assert set(model_from_str.edges()) == expected_edges

    def test_from_dagitty_DAG_ctor(self):
        from pgmpy.base import DAG

        # Adapted from https://www.dagitty.net/manual-3.x.pdf#page=4 section 3.1 with beta modified
        model_str = """dag{
        "carry matches" [latent]
        cancer [outcome]
        smoking -> "carry matches" [beta=0.2]
        smoking -> cancer [beta=0.5]
        "carry matches" -> cancer
        }"""
        model_from_str = DAG.from_dagitty(model_str)
        assert isinstance(model_from_str, LinearGaussianBayesianNetwork)
        assert sorted(model_from_str.nodes()) == ["cancer", "carry matches", "smoking"]
        expected_edges = {
            ("smoking", "carry matches"),
            ("smoking", "cancer"),
            ("carry matches", "cancer"),
        }
        assert set(model_from_str.edges()) == expected_edges
        assert model_from_str.check_model() == True

        # Test CPDs
        assert len(model_from_str.cpds) == 3

        # Check if all std dev are set
        assert model_from_str.get_cpds("cancer").std is not None
        assert model_from_str.get_cpds("carry matches").std is not None
        assert model_from_str.get_cpds("smoking").std is not None

        # Check variable names
        assert model_from_str.get_cpds("cancer").variable == "cancer"
        assert model_from_str.get_cpds("carry matches").variable == "carry matches"
        assert model_from_str.get_cpds("smoking").variable == "smoking"

        # Check evidences
        assert sorted(model_from_str.get_cpds("cancer").evidence) == ["carry matches", "smoking"]
        assert sorted(model_from_str.get_cpds("carry matches").evidence) == ["smoking"]
        assert sorted(model_from_str.get_cpds("smoking").evidence) == []

        # Check if the betas specified were correctly set
        assert model_from_str.get_cpds("cancer").beta[1] == 0.5
        assert model_from_str.get_cpds("carry matches").beta[1] == 0.2

        # Check if intercepts are 0
        assert model_from_str.get_cpds("cancer").beta[0] == 0.0
        assert model_from_str.get_cpds("carry matches").beta[0] == 0.0
        assert model_from_str.get_cpds("smoking").beta[0] == 0.0

        # Check if std devs are 1
        assert model_from_str.get_cpds("cancer").std == 1
        assert model_from_str.get_cpds("carry matches").std == 1
        assert model_from_str.get_cpds("smoking").std == 1


class TestLGBNIO:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = load_model("bnlearn/ecoli70")
        self.filename = "ecoli70.json"
        self.model.save(self.filename)
        yield
        # Clean up the test file
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_save_and_load(self):
        """Test basic save and load functionality"""
        loaded_model = LinearGaussianBayesianNetwork.load("ecoli70.json")
        assert self.model == loaded_model

    def test_load_from_file_object(self):
        """Test loading from file-like object"""
        with open(self.filename, "rb") as f:
            file_obj = io.BytesIO(f.read())

        loaded_model = LinearGaussianBayesianNetwork.load(file_obj)
        assert self.model == loaded_model

    def test_simulate_missing_prob(self):
        model = LinearGaussianBayesianNetwork([("X1", "X2")])
        cpd1 = LinearGaussianCPD("X1", [0], 1)
        cpd2 = LinearGaussianCPD("X2", [0, 1], 1, evidence=["X1"])

        model.add_cpds(cpd1, cpd2)
        df = model.simulate(n_samples=500, missing_prob={"X1": 0.5})
        nan_ratio = df["X1"].isna().mean()

        assert 0.4 <= nan_ratio <= 0.6

    def test_simulate_missing_prob_invalid_node(self):
        model = LinearGaussianBayesianNetwork([("X1", "X2")])
        cpd1 = LinearGaussianCPD("X1", [0], 1)
        cpd2 = LinearGaussianCPD("X2", [0, 1], 1, evidence=["X1"])
        model.add_cpds(cpd1, cpd2)

        with pytest.raises(ValueError):
            model.simulate(n_samples=10, missing_prob={"X3": 0.5})

    def test_simulate_missing_prob_invalid_probability(self):
        model = LinearGaussianBayesianNetwork([("X1", "X2")])
        cpd1 = LinearGaussianCPD("X1", [0], 1)
        cpd2 = LinearGaussianCPD("X2", [0, 1], 1, evidence=["X1"])
        model.add_cpds(cpd1, cpd2)

        with pytest.raises(ValueError):
            model.simulate(n_samples=10, missing_prob={"X1": -0.1})

        with pytest.raises(ValueError):
            model.simulate(n_samples=10, missing_prob={"X1": 1.5})
