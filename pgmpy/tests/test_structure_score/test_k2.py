import pytest

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.structure_score import K2


class TestK2:
    def test_score(self, small_df, small_df_models):
        m1, m2 = small_df_models
        scorer = K2(small_df)
        assert scorer.score(m1) == pytest.approx(-10.73813429536977)
        assert scorer.score(m2) == pytest.approx(-10.345091707260167)
        assert scorer.score(DiscreteBayesianNetwork()) == 0

    def test_score_titanic(self, titanic_data):
        scorer = K2(titanic_data)
        titanic = DiscreteBayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        assert scorer.score(titanic) == pytest.approx(-1891.0630673606006)

        titanic2 = DiscreteBayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        assert scorer.score(titanic2) < scorer.score(titanic)
