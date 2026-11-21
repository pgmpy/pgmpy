import numpy as np
import pytest

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.structure_score import BDeu, BDs

# Score values in the tests are compared to R package bnlearn


class TestBDeu:
    def test_score(self, small_df, small_df_models):
        m1, m2 = small_df_models
        scorer = BDeu(small_df)
        assert scorer.score(m1) == pytest.approx(-9.907103407446435)
        assert scorer.score(m2) == pytest.approx(-9.839964104608821)
        assert scorer.score(DiscreteBayesianNetwork()) == 0

    def test_score_titanic(self, titanic_data):
        scorer = BDeu(titanic_data, equivalent_sample_size=25)
        titanic = DiscreteBayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        assert scorer.score(titanic) == pytest.approx(-1892.7383393910427)

        titanic2 = DiscreteBayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        assert scorer.score(titanic2) < scorer.score(titanic)

    @pytest.mark.parametrize("scorer_class", [BDeu, BDs])
    @pytest.mark.parametrize("equivalent_sample_size", [0, -1, -0.5, None, "10", np.nan, np.inf])
    def test_invalid_equivalent_sample_size(self, small_df, scorer_class, equivalent_sample_size):
        with pytest.raises(ValueError, match="equivalent_sample_size must be a positive finite number"):
            scorer_class(small_df, equivalent_sample_size=equivalent_sample_size)

    @pytest.mark.parametrize("scorer_class", [BDeu, BDs])
    def test_fractional_equivalent_sample_size(self, small_df, small_df_models, scorer_class):
        scorer = scorer_class(small_df, equivalent_sample_size=0.5)
        assert np.isfinite(scorer.score(small_df_models[0]))
