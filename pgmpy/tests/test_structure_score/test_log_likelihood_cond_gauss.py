import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from pgmpy.structure_score import LogLikelihoodCondGauss


class TestLogLikelihoodCondGauss:
    def test_score_manual_continuous(self, loglik_cond_gauss_manual_score):
        # bnlearn 5.0.2, score(dag, d[1:2, ], type = "loglik-cg", by.node = TRUE)["A"] -> -Inf for both families
        assert loglik_cond_gauss_manual_score.local_score(variable="A", parents=("B_cat",)) == -np.inf
        assert loglik_cond_gauss_manual_score.local_score(variable="A", parents=("B_cat", "B")) == -np.inf

    def test_score_manual_categorical(self, loglik_cond_gauss_manual_score):
        assert loglik_cond_gauss_manual_score.local_score(variable="A_cat", parents=("B",)) == -np.inf
        assert loglik_cond_gauss_manual_score.local_score(variable="A_cat", parents=("B_cat", "A")) == -np.inf
        assert (
            loglik_cond_gauss_manual_score.local_score(variable="A_cat", parents=("B", "B_cat", "C", "C_cat"))
            == -np.inf
        )

    def test_score_bnlearn_no_parents(self, loglik_cond_gauss_score):
        # R 4.4.3, logLik(lm(X ~ 1, d)): A -> -119.7202, B -> -257.0042, C -> -328.2361. bnlearn 5.0.2 plugs the
        # residual standard error, RSS / (n - p), into the likelihood instead of the MLE, so its by.node values
        # differ by about p^2 / (4n), e.g. A -> -119.7228.
        assert loglik_cond_gauss_score.local_score(variable="A", parents=()) == pytest.approx(-119.7202, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="B", parents=()) == pytest.approx(-257.0042, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="B_cat", parents=()) == pytest.approx(-81.6952, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="C", parents=()) == pytest.approx(-328.2361, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="C_cat", parents=()) == pytest.approx(-130.1208, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=()) == pytest.approx(-121.527, abs=1e-3)

    def test_score_bnlearn_continuous_parent(self, loglik_cond_gauss_score):
        # R 4.4.3, logLik(lm(A ~ B, d)) -> -119.4910; bnlearn 5.0.2 by.node loglik-g -> -119.5012
        assert loglik_cond_gauss_score.local_score(variable="A", parents=("B",)) == pytest.approx(-119.4910, abs=1e-3)

    def test_score_bnlearn_categorical_parent(self, loglik_cond_gauss_score):
        # R 4.4.3, sum of logLik(lm(A ~ 1, d_k)) over the B_cat strata d_k -> -118.4768; bnlearn 5.0.2 -> -118.5250
        assert loglik_cond_gauss_score.local_score(variable="A", parents=("B_cat",)) == pytest.approx(
            -118.4768, abs=1e-3
        )
        # score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat",)) == pytest.approx(
            -117.6219, abs=1e-3
        )

    def test_score_bnlearn_mixed_parents(self, loglik_cond_gauss_score):
        # R 4.4.3, sum of logLik(lm(A ~ B, d_k)) over the B_cat strata d_k -> -113.1890; bnlearn 5.0.2 -> -113.3972
        assert loglik_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B")) == pytest.approx(
            -113.1890, abs=1e-3
        )

    def test_score_bnlearn_many_parents(self, loglik_cond_gauss_score):
        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), type='loglik-cg') -> -Inf
        assert loglik_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B", "C_cat", "C")) == -np.inf

    def test_score_bnlearn_continuous_to_categorical(self, loglik_cond_gauss_score):
        # bnlearn does not allow continuous parents of a discrete node. scipy 1.17.0: sum over the A_cat strata of
        # multivariate_normal.logpdf of B with ddof=0 moments plus n_k log(n_k / n), minus the same for all of B
        # -> -116.6293
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=("B",)) == pytest.approx(
            -116.6293, abs=1e-3
        )
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat", "A")) == -np.inf
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=("B", "B_cat", "C", "C_cat")) == -np.inf

    def test_score_is_not_floored_for_low_density_rows(self):
        rng = np.random.default_rng(seed=0)
        df = pd.DataFrame(rng.normal(size=(200, 4)), columns=["A", "B", "C", "D"])
        df.loc[0, ["B", "C", "D"]] = [12.0, -12.0, 12.0]

        def gaussian_log_likelihood(frame):
            return multivariate_normal.logpdf(frame, mean=frame.mean(), cov=frame.cov(ddof=0)).sum()

        expected = gaussian_log_likelihood(df) - gaussian_log_likelihood(df[["B", "C", "D"]])
        assert LogLikelihoodCondGauss(df).local_score("A", ("B", "C", "D")) == pytest.approx(expected, abs=1e-6)

    def test_degenerate_families_score_minus_inf(self):
        rng = np.random.default_rng(seed=0)
        df = pd.DataFrame({"A": rng.normal(size=60), "B": rng.normal(size=60)})
        df["K2"] = np.where(df.index < 2, "pair", "rest")
        df["K3"] = np.where(df.index < 3, "trio", "rest")
        score = LogLikelihoodCondGauss(df)

        # bnlearn 5.0.2, score(dag, df, type = "loglik-cg", by.node = TRUE)[child]: A | B, K2 -> -Inf;
        # A | B, K3 -> -69.62216
        assert score.local_score("A", ("B", "K2")) == -np.inf
        assert np.isfinite(score.local_score("A", ("B", "K3")))

        assert score.local_score("K2", ("A", "B")) == -np.inf
        assert np.isfinite(score.local_score("K3", ("A", "B")))

    @pytest.mark.parametrize(
        ("variable", "parents"), [("A", ("B",)), ("A", ("B", "C")), ("A", ("B", "G")), ("G", ("A", "B"))]
    )
    def test_rescaling_shifts_score_by_jacobian(self, variable, parents):
        rng = np.random.default_rng(seed=3)
        df = pd.DataFrame({"A": rng.normal(size=120), "B": rng.normal(size=120), "C": rng.normal(size=120)})
        df["G"] = rng.choice(["p", "q"], size=120)
        scale = 1e-5
        scaled = df.assign(**{col: df[col] * scale for col in ("A", "B", "C")})

        shift = -len(df) * np.log(scale) if variable == "A" else 0.0
        expected = LogLikelihoodCondGauss(df).local_score(variable, parents)
        assert LogLikelihoodCondGauss(scaled).local_score(variable, parents) == pytest.approx(
            expected + shift, rel=1e-9
        )

        rescaled_parent = df.assign(B=df["B"] * 1e-8)
        assert LogLikelihoodCondGauss(rescaled_parent).local_score(variable, parents) == pytest.approx(
            expected, rel=1e-9
        )

    def test_bool_is_categorical_in_mixed_data(self):
        rng = np.random.default_rng(seed=4)
        data = pd.DataFrame({"A": rng.normal(size=80), "C": rng.normal(size=80), "G": rng.random(80) < 0.5})
        data["A"] += data["G"]
        data["H"] = rng.choice(["x", "y"], size=80)
        as_category = data.assign(G=data["G"].astype("category"))

        for variable, parents in [("A", ("G",)), ("G", ("A",)), ("A", ("C", "G"))]:
            assert LogLikelihoodCondGauss(data).local_score(variable, parents) == pytest.approx(
                LogLikelihoodCondGauss(as_category).local_score(variable, parents), rel=1e-12
            )
