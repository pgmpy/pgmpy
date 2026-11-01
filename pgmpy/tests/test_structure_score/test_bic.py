import pytest

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.structure_score import BIC


class TestBIC:
    def test_score(self, small_df, small_df_models):
        m1, m2 = small_df_models
        scorer = BIC(small_df)
        assert scorer.score(m1) == pytest.approx(-10.698440814229318)
        assert scorer.score(m2) == pytest.approx(-9.625886526130714)
        assert scorer.score(DiscreteBayesianNetwork()) == 0

    def test_score_titanic(self, titanic_data):
        scorer = BIC(titanic_data)
        titanic = DiscreteBayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        assert scorer.score(titanic) == pytest.approx(-1896.7250012840179)

        titanic2 = DiscreteBayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        assert scorer.score(titanic2) < scorer.score(titanic)
