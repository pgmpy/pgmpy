import pytest


class TestBICGauss:
    def test_local_score_no_parents(self, bic_gauss_score):
        assert bic_gauss_score.local_score(variable="A", parents=()) == pytest.approx(-124.3254, abs=1e-3)
        assert bic_gauss_score.local_score(variable="B", parents=()) == pytest.approx(-261.6093, abs=1e-3)

    def test_local_score_with_parents(self, bic_gauss_score):
        assert bic_gauss_score.local_score(variable="C", parents=("A", "B")) == pytest.approx(-87.5918, abs=1e-3)

    def test_score(self, bic_gauss_score, gauss_models):
        m1, m2 = gauss_models
        assert bic_gauss_score.score(m1) == pytest.approx(-473.5265, abs=1e-3)
        assert bic_gauss_score.score(m2) == pytest.approx(-587.8711, abs=1e-3)
