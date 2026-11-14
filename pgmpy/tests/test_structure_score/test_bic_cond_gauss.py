import numpy as np
import pytest


class TestBICCondGauss:
    def test_score_bnlearn_no_parents(self, bic_cond_gauss_score):
        assert bic_cond_gauss_score.local_score(variable="B_cat", parents=()) == pytest.approx(-86.3004, abs=1e-3)
        # R 4.4.3 log-likelihoods B -> -257.0042 and C -> -328.2361, minus 2 log(100) / 2
        assert bic_cond_gauss_score.local_score(variable="B", parents=()) == pytest.approx(-261.6093, abs=1e-3)
        assert bic_cond_gauss_score.local_score(variable="C", parents=()) == pytest.approx(-332.8412, abs=1e-3)
        assert bic_cond_gauss_score.local_score(variable="C_cat", parents=()) == pytest.approx(-139.3311, abs=1e-3)
        # score(model2network("[A_cat]"), d[c('A_cat')], type='loglik') -> -121.527
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=()) == pytest.approx(-128.4347, abs=1e-3)

    def test_score_bnlearn_categorical_parent(self, bic_cond_gauss_score):
        # R 4.4.3 log-likelihood -118.4768 minus 6 log(100) / 2
        assert bic_cond_gauss_score.local_score(variable="A", parents=("B_cat",)) == pytest.approx(-132.2923, abs=1e-3)
        # score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat",)) == pytest.approx(
            -138.3452, abs=1e-3
        )

    def test_score_bnlearn_mixed_parents(self, bic_cond_gauss_score):
        # R 4.4.3 log-likelihood -113.1890 minus 9 log(100) / 2
        assert bic_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B")) == pytest.approx(
            -133.9123, abs=1e-3
        )

    def test_score_bnlearn_many_parents(self, bic_cond_gauss_score):
        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), type='loglik-cg') -> -Inf
        assert bic_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B", "C_cat", "C")) == -np.inf

    def test_score_bnlearn_continuous_to_categorical(self, bic_cond_gauss_score):
        # bnlearn does not allow continuous parents of a discrete node; scipy 1.17.0 log-likelihood -116.6293
        # minus 9 log(100) / 2
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=("B",)) == pytest.approx(-137.3526, abs=1e-3)
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat", "A")) == -np.inf
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=("B", "B_cat", "C", "C_cat")) == -np.inf
