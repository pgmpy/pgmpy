import pytest

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.structure_score import AIC


class TestAIC:
    def test_score(self, small_df, small_df_models):
        m1, m2 = small_df_models
        scorer = AIC(small_df)
        assert scorer.score(m1) == pytest.approx(-15.205379370888767)
        assert scorer.score(m2) == pytest.approx(-13.68213122712422)
        assert scorer.score(DiscreteBayesianNetwork()) == 0

    def test_score_titanic(self, titanic_data):
        scorer = AIC(titanic_data)
        titanic = DiscreteBayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        assert scorer.score(titanic) == pytest.approx(-1875.1594513603993)

        titanic2 = DiscreteBayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        assert scorer.score(titanic2) < scorer.score(titanic)
