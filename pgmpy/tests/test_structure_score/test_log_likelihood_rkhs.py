import pytest


class TestLogLikelihoodRKHS:
    def test_local_score_no_parents(self, loglik_rkhs_score):
        assert loglik_rkhs_score.local_score(variable="A", parents=()) == pytest.approx(-372.369, abs=1e-3)
        assert loglik_rkhs_score.local_score(variable="B", parents=()) == pytest.approx(-372.547, abs=1e-3)
        assert loglik_rkhs_score.local_score(variable="C", parents=()) == pytest.approx(-372.596, abs=1e-3)

    def test_local_score_with_parents(self, loglik_rkhs_score):
        assert loglik_rkhs_score.local_score(variable="C", parents=("A", "B")) == pytest.approx(100967.845, abs=1e-3)
        assert loglik_rkhs_score.local_score(variable="D", parents=("A", "B", "C")) == pytest.approx(
            116442.641, abs=1e-3
        )

    def test_score(self, loglik_rkhs_score, gauss_models):
        m1, m2 = gauss_models
        assert loglik_rkhs_score.score(m1) == pytest.approx(100222.929, abs=1e-3)
        assert loglik_rkhs_score.score(m2) == pytest.approx(229249.447, abs=1e-3)
