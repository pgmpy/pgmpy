import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from pgmpy.structure_score import LogLikelihoodCondGauss


class TestLogLikelihoodCondGauss:
    def test_score_manual_continuous(self, loglik_cond_gauss_manual_score):
        assert loglik_cond_gauss_manual_score.local_score(variable="A", parents=("B_cat",)) == pytest.approx(
            -1.8378, abs=1e-3
        )
        assert loglik_cond_gauss_manual_score.local_score(variable="A", parents=("B_cat", "B")) == pytest.approx(
            -1.8379, abs=1e-3
        )

    def test_score_manual_categorical(self, loglik_cond_gauss_manual_score):
        assert loglik_cond_gauss_manual_score.local_score(variable="A_cat", parents=("B",)) == pytest.approx(
            2.9024, abs=1e-3
        )
        assert loglik_cond_gauss_manual_score.local_score(variable="A_cat", parents=("B_cat", "A")) == pytest.approx(
            0, abs=1e-3
        )
        assert loglik_cond_gauss_manual_score.local_score(
            variable="A_cat", parents=("B", "B_cat", "C", "C_cat")
        ) == pytest.approx(0, abs=1e-3)

    def test_score_bnlearn_no_parents(self, loglik_cond_gauss_score):
        # score(model2network("[A]"), d[c('A')], type='loglik-g') -> -119.7228
        assert loglik_cond_gauss_score.local_score(variable="A", parents=()) == pytest.approx(-119.7228, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="B", parents=()) == pytest.approx(-257.0067, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="B_cat", parents=()) == pytest.approx(-81.6952, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="C", parents=()) == pytest.approx(-328.2386, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="C_cat", parents=()) == pytest.approx(-130.1208, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=()) == pytest.approx(-121.527, abs=1e-3)

    def test_score_bnlearn_continuous_parent(self, loglik_cond_gauss_score):
        # score(model2network("[B][A|B]"), d[c('A', 'B')], type='loglik-g') -> 376.5078
        assert loglik_cond_gauss_score.local_score(variable="A", parents=("B",)) == pytest.approx(-119.4935, abs=1e-3)

    def test_score_bnlearn_categorical_parent(self, loglik_cond_gauss_score):
        # score(model2network("[B_cat][A|B_cat]"), d[c('A', 'B_cat')], type='loglik-cg') -> 200.2201
        assert loglik_cond_gauss_score.local_score(variable="A", parents=("B_cat",)) == pytest.approx(
            -118.5250, abs=1e-3
        )
        # score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat",)) == pytest.approx(
            -117.6219, abs=1e-3
        )

    def test_score_bnlearn_mixed_parents(self, loglik_cond_gauss_score):
        # score(model2network("[B][B_cat][A|B:B_cat]"), d[c('A', 'B', 'B_cat')], type='loglik-cg') -> 452.0991
        assert loglik_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B")) == pytest.approx(
            -113.2371, abs=1e-3
        )

    def test_score_bnlearn_many_parents(self, loglik_cond_gauss_score):
        # Strata with fewer rows than dimensions: the value reflects how `_adjusted_cov` regularizes them.
        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), type='loglik-cg') -> -Inf
        assert loglik_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B", "C_cat", "C")) == pytest.approx(
            15.2256, abs=1e-3
        )

    def test_score_bnlearn_continuous_to_categorical(self, loglik_cond_gauss_score):
        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=("B",)) == pytest.approx(
            -116.7104, abs=1e-3
        )
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat", "A")) == pytest.approx(
            -6.1599, abs=1e-3
        )
        assert loglik_cond_gauss_score.local_score(
            variable="A_cat", parents=("B", "B_cat", "C", "C_cat")
        ) == pytest.approx(31.6170, abs=1e-3)

    def test_score_is_not_floored_for_low_density_rows(self):
        rng = np.random.default_rng(seed=0)
        df = pd.DataFrame(rng.normal(size=(200, 4)), columns=["A", "B", "C", "D"])
        df.loc[0, ["B", "C", "D"]] = [12.0, -12.0, 12.0]

        def gaussian_log_likelihood(frame):
            return multivariate_normal.logpdf(frame, mean=frame.mean(), cov=frame.cov()).sum()

        expected = gaussian_log_likelihood(df) - gaussian_log_likelihood(df[["B", "C", "D"]])
        assert LogLikelihoodCondGauss(df).local_score("A", ("B", "C", "D")) == pytest.approx(expected, abs=1e-6)

    def test_adjusted_cov_regularizes_duplicated_column(self):
        df = pd.DataFrame({"A": np.random.default_rng(seed=0).normal(size=50)})
        df["B"] = df["A"]

        cov, is_sample_cov = LogLikelihoodCondGauss._adjusted_cov(df)
        assert not is_sample_cov
        assert np.linalg.eigvalsh(cov.to_numpy())[0] > 0

    @pytest.mark.parametrize(
        ("variable", "parents"), [("A", ("B",)), ("A", ("B", "C")), ("A", ("B", "G")), ("G", ("A", "B"))]
    )
    def test_rescaling_shifts_score_by_jacobian(self, variable, parents):
        rng = np.random.default_rng(seed=3)
        df = pd.DataFrame({"A": rng.normal(size=120), "B": rng.normal(size=120), "C": rng.normal(size=120)})
        df["G"] = rng.choice(["p", "q"], size=120)
        scale = 1e-5
        scaled = df.assign(**{col: df[col] * scale for col in ("A", "B", "C")})

        # Each density of a continuous child is multiplied by 1 / scale; for a discrete child the Jacobians cancel.
        shift = -len(df) * np.log(scale) if variable == "A" else 0.0
        expected = LogLikelihoodCondGauss(df).local_score(variable, parents) + shift
        assert LogLikelihoodCondGauss(scaled).local_score(variable, parents) == pytest.approx(expected, rel=1e-9)
