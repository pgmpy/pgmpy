import pandas as pd
import pytest

from pgmpy.structure_score import BDs


class TestBDs:
    def test_score(self, bds_df, bds_models):
        m1, m2 = bds_models
        scorer = BDs(bds_df, equivalent_sample_size=1)
        assert scorer.score(m1) == pytest.approx(-36.82311976667139)
        assert scorer.score(m2) == pytest.approx(-37.51626694723133)

    def test_local_score_unobserved_parent_configs(self, bds_df):
        # Example 1 of Scutari (2018) and of Scutari (2016); expected values are from bnlearn.
        scorer = BDs(bds_df, equivalent_sample_size=1)
        assert scorer.local_score("X", ("W", "Z", "Y")) == pytest.approx(-3.4226644402308803)
        assert scorer.local_score("X", ("W", "Z", "Y")) == pytest.approx(scorer.local_score("X", ("W", "Z")))

        data = pd.DataFrame({"X1": [2, 2, 2, 2, 2, 2, 2], "X2": [1, 1, 2, 2, 2, 2, 2]})
        scorer = BDs(data, equivalent_sample_size=1, state_names={"X1": [1, 2], "X2": [1, 2]})
        assert scorer.local_score("X2", ("X1",)) == pytest.approx(-5.4273944088231794)
