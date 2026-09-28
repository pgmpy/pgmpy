import numpy as np
import pandas as pd
import pytest

from pgmpy.structure_score import LogLikelihoodGauss


@pytest.fixture
def collinear_data():
    rng = np.random.default_rng(seed=1)
    df = pd.DataFrame({"A": rng.normal(size=300), "B": rng.normal(size=300)})
    df["C"] = df["A"] + df["B"]
    df["D"] = df["A"] + rng.normal(size=300)
    return df


class TestLogLikeGauss:
    def test_local_score_no_parents(self, loglik_gauss_score):
        # score(model2network("[A]"), df[c('A')], type='loglik-g') -> -119.7228
        assert loglik_gauss_score.local_score(variable="A", parents=()) == pytest.approx(-119.7202, abs=1e-3)

        # score(model2network("[B]"), df[c('B')], type='loglik-g') -> -257.0067
        assert loglik_gauss_score.local_score(variable="B", parents=()) == pytest.approx(-257.0042, abs=1e-3)

        # score(model2network("[C]"), df[c('C')], type='loglik-g')
        assert loglik_gauss_score.local_score(variable="C", parents=()) == pytest.approx(-328.2361, abs=1e-3)

    def test_local_score_with_parents(self, loglik_gauss_score):
        # score(model2network("[A][B][C|A:B]"), df[c('A', 'B', 'C')], type='loglik-g') -> -455.1339
        assert loglik_gauss_score.local_score(variable="C", parents=("A", "B")) == pytest.approx(-78.3815, abs=1e-3)

        # score(model2network("[A][B][C][D|A:B:C]"), df[c('A', 'B', 'C', 'D')], type='loglik-g') -> -732.2027
        assert loglik_gauss_score.local_score(variable="D", parents=("A", "B", "C")) == pytest.approx(
            -27.1936, abs=1e-3
        )

    def test_score(self, loglik_gauss_score, gauss_models):
        m1, m2 = gauss_models
        assert loglik_gauss_score.score(m1) == pytest.approx(-455.1058, abs=1e-3)
        assert loglik_gauss_score.score(m2) == pytest.approx(-569.4505, abs=1e-3)

    def test_missing_values_raise_clear_error(self, collinear_data):
        collinear_data.loc[0, "B"] = np.nan
        score = LogLikelihoodGauss(collinear_data)

        with pytest.raises(ValueError, match=r"missing values.*B"):
            score.local_score("A", ("B",))
        with pytest.raises(ValueError, match=r"missing values.*B"):
            score.local_score("B", ())

        # A variable with missing values that is not part of the family must not interfere.
        assert np.isfinite(score.local_score("A", ("D",)))

    def test_collinear_parents_and_determined_variable(self, collinear_data):
        score = LogLikelihoodGauss(collinear_data)

        # C = A + B: a redundant parent is ignored, but a family its parents determine scores -inf.
        assert np.isfinite(score.local_score("D", ("A", "B", "C")))
        assert score.local_score("D", ("A", "B", "C")) == pytest.approx(score.local_score("D", ("A", "B")), abs=1e-8)
        assert score.local_score("C", ("A", "B")) == -np.inf
        assert score.local_score("A", ("B", "C")) == -np.inf
