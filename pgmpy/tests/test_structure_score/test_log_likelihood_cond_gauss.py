import pytest


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
        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), type='loglik-cg') -> -Inf
        assert loglik_cond_gauss_score.local_score(variable="A", parents=("B_cat", "B", "C_cat", "C")) == pytest.approx(
            19.1557, abs=1e-3
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
        ) == pytest.approx(41.9122, abs=1e-3)
