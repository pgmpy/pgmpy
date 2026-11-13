import pytest


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
