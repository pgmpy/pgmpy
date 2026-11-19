import pytest


class TestAICCondGauss:
    def test_score_bnlearn_no_parents(self, aic_cond_gauss_score):
        assert aic_cond_gauss_score.local_score(variable="B_cat", parents=()) == pytest.approx(-83.6952, abs=1e-3)
        assert aic_cond_gauss_score.local_score(variable="B", parents=()) == pytest.approx(-259.0067, abs=1e-3)
        assert aic_cond_gauss_score.local_score(variable="C", parents=()) == pytest.approx(-330.2386, abs=1e-3)
        assert aic_cond_gauss_score.local_score(variable="C_cat", parents=()) == pytest.approx(-134.1208, abs=1e-3)
        # score(model2network("[A_cat]"), d[c('A_cat')], type='loglik') -> -121.527
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=()) == pytest.approx(-124.527, abs=1e-3)

    def test_score_bnlearn_categorical_parent(self, aic_cond_gauss_score):
        # score(model2network("[B_cat][A|B_cat]"), d[c('A', 'B_cat')], type='aic-cg') -> 208.2201
        assert aic_cond_gauss_score.local_score(variable="A", parents=("B_cat",)) == pytest.approx(-124.525, abs=1e-3)
        # score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat",)) == pytest.approx(
            -126.6219, abs=1e-3
        )

    def test_score_bnlearn_mixed_parents(self, aic_cond_gauss_score):
        # score(model2network("[B][B_cat][A|B:B_cat]"), d[c('A', 'B', 'B_cat')], type='loglik-cg') -> 465.0991
        assert aic_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B")) == pytest.approx(
            -122.2372, abs=1e-3
        )

    def test_score_bnlearn_many_parents(self, aic_cond_gauss_score):
        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), type='loglik-cg') -> -Inf
        assert aic_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B", "C_cat", "C")) == pytest.approx(
            -40.8443, abs=1e-3
        )

    def test_score_bnlearn_continuous_to_categorical(self, aic_cond_gauss_score):
        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=("B",)) == pytest.approx(-125.7104, abs=1e-3)
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat", "A")) == pytest.approx(
            -33.1599, abs=1e-3
        )
        assert aic_cond_gauss_score.local_score(
            variable="A_cat", parents=("B", "B_cat", "C", "C_cat")
        ) == pytest.approx(-138.0878, abs=1e-3)
