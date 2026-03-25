import pytest


class TestAICGauss:
    def test_local_score_no_parents(self, aic_gauss_score):
        # score(model2network("[A]"), df_cont[c('A')], type='aic-g') -> -121.7228
        assert aic_gauss_score.local_score(variable="A", parents=()) == pytest.approx(-121.7202, abs=1e-3)

        # score(model2network("[B]"), df_cont[c('B')], type='aic-g') -> -259.0067
        assert aic_gauss_score.local_score(variable="B", parents=()) == pytest.approx(-259.0042, abs=1e-3)

        # score(model2network("[C]"), df_cont[c('C')], type='aic-g') -> -330.2386
        assert aic_gauss_score.local_score(variable="C", parents=()) == pytest.approx(-330.2361, abs=1e-3)

    def test_local_score_with_parents(self, aic_gauss_score):
        # score(model2network("[A][B][C|A:B]"), df_cont[c('A', 'B', 'C')], type='aic-g') -> -463.1339
        assert aic_gauss_score.local_score(variable="C", parents=("A", "B")) == pytest.approx(-82.3815, abs=1e-3)

        # score(model2network("[A][B][C][D|A:B:C]"), df_cont[c('A', 'B', 'C', 'D')], type='aic-g')
        assert aic_gauss_score.local_score(variable="D", parents=("A", "B", "C")) == pytest.approx(-32.1936, abs=1e-3)

    def test_score(self, aic_gauss_score, gauss_models):
        m1, m2 = gauss_models
        assert aic_gauss_score.score(m1) == pytest.approx(-463.1059, abs=1e-3)
        assert aic_gauss_score.score(m2) == pytest.approx(-577.4505, abs=1e-3)
