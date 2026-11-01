import pytest

from pgmpy.structure_score import BDs


class TestBDs:
    def test_score(self, bds_df, bds_models):
        m1, m2 = bds_models
        scorer = BDs(bds_df, equivalent_sample_size=1)
        assert scorer.score(m1) == pytest.approx(-36.82311976667139)
        assert scorer.score(m2) == pytest.approx(-45.788991276221964)
