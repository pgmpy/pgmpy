import pytest


class TestBICCondGauss:
    def test_score_bnlearn_no_parents(self, bic_cond_gauss_score):
        assert bic_cond_gauss_score.local_score(variable="B_cat", parents=()) == pytest.approx(-86.3004, abs=1e-3)
        assert bic_cond_gauss_score.local_score(variable="B", parents=()) == pytest.approx(-261.6119, abs=1e-3)
        assert bic_cond_gauss_score.local_score(variable="C", parents=()) == pytest.approx(-332.8438, abs=1e-3)
        assert bic_cond_gauss_score.local_score(variable="C_cat", parents=()) == pytest.approx(-139.3311, abs=1e-3)
        # score(model2network("[A_cat]"), d[c('A_cat')], type='loglik') -> -121.527
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=()) == pytest.approx(-128.4347, abs=1e-3)

    def test_score_bnlearn_categorical_parent(self, bic_cond_gauss_score):
        # score(model2network("[B_cat][A|B_cat]"), d[c('A', 'B_cat')], type='bic-cg') -> 218.6408
        assert bic_cond_gauss_score.local_score(variable="A", parents=("B_cat",)) == pytest.approx(-132.3405, abs=1e-3)
        # score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat",)) == pytest.approx(
            -138.3452, abs=1e-3
        )

    def test_score_bnlearn_mixed_parents(self, bic_cond_gauss_score):
        # score(model2network("[B][B_cat][A|B:B_cat]"), d[c('A', 'B', 'B_cat')], type='loglik-cg') -> 482.0327
        assert bic_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B")) == pytest.approx(
            -133.9605, abs=1e-3
        )

    def test_score_bnlearn_many_parents(self, bic_cond_gauss_score):
        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), type='loglik-cg') -> -Inf
        assert bic_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B", "C_cat", "C")) == pytest.approx(
            -118.9994, abs=1e-3
        )

    def test_score_bnlearn_continuous_to_categorical(self, bic_cond_gauss_score):
        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=("B",)) == pytest.approx(-137.4337, abs=1e-3)
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=("B_cat", "A")) == pytest.approx(
            -68.3297, abs=1e-3
        )
        assert bic_cond_gauss_score.local_score(
            variable="A_cat", parents=("B", "B_cat", "C", "C_cat")
        ) == pytest.approx(-372.5531, abs=1e-3)
