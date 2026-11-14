import numpy as np
import pytest


class TestAICCondGauss:
    def test_score_bnlearn_no_parents(self, aic_cond_gauss_score):
        assert aic_cond_gauss_score.local_score(variable="B_cat", parents=()) == pytest.approx(-83.6952, abs=1e-3)
        # R 4.4.3 log-likelihoods B -> -257.0042 and C -> -328.2361, minus 2 parameters
        assert aic_cond_gauss_score.local_score(variable="B", parents=()) == pytest.approx(-259.0042, abs=1e-3)
        assert aic_cond_gauss_score.local_score(variable="C", parents=()) == pytest.approx(-330.2361, abs=1e-3)
        assert aic_cond_gauss_score.local_score(variable="C_cat", parents=()) == pytest.approx(-134.1208, abs=1e-3)
        # score(model2network("[A_cat]"), d[c('A_cat')], type='loglik') -> -121.527
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=()) == pytest.approx(-124.527, abs=1e-3)

    def test_score_bnlearn_categorical_parent(self, aic_cond_gauss_score):
        # R 4.4.3 log-likelihood -118.4768 minus 6 parameters
        assert aic_cond_gauss_score.local_score(variable="A", parents=("B_cat",)) == pytest.approx(-124.4768, abs=1e-3)
        # score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat",)) == pytest.approx(
            -126.6219, abs=1e-3
        )

    def test_score_bnlearn_mixed_parents(self, aic_cond_gauss_score):
        # R 4.4.3 log-likelihood -113.1890 minus 9 parameters
        assert aic_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B")) == pytest.approx(
            -122.1890, abs=1e-3
        )

    def test_score_bnlearn_many_parents(self, aic_cond_gauss_score):
        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), type='loglik-cg') -> -Inf
        assert aic_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B", "C_cat", "C")) == -np.inf

    def test_score_bnlearn_continuous_to_categorical(self, aic_cond_gauss_score):
        # bnlearn does not allow continuous parents of a discrete node; scipy 1.17.0 log-likelihood -116.6293
        # minus 9 parameters
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=("B",)) == pytest.approx(-125.6293, abs=1e-3)
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat", "A")) == -np.inf
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=("B", "B_cat", "C", "C_cat")) == -np.inf
