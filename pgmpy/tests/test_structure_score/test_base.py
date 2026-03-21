import pandas as pd
import pytest

from pgmpy.structure_score import (
    BIC,
    K2,
    BaseStructureScore,
    BICCondGauss,
    BICGauss,
)
from pgmpy.structure_score._base import get_scoring_method


class TestBaseStructureScore:
    def test_default_for_tag(self):
        assert BaseStructureScore.get_class_tag("default_for") is None

    def test_local_score_requires_tuple_parents(self, small_df):
        score = K2(small_df.astype("category"))

        with pytest.raises(TypeError, match=r"`parents` must be a tuple\."):
            score.local_score("A", ["B"])


class TestGetScoringMethod:
    def test_get_scoring_method_default_discrete(self, small_df):
        data = small_df.astype("category")
        score, score_c = get_scoring_method(None, data, use_cache=False)

        assert isinstance(score, BIC)
        assert score_c is score

    def test_get_scoring_method_default_continuous(self):
        data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")

        score, score_c = get_scoring_method(None, data, use_cache=False)

        assert isinstance(score, BICGauss)
        assert score_c is score

    def test_get_scoring_method_default_mixed(self):
        data = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)

        score, score_c = get_scoring_method(None, data, use_cache=False)

        assert isinstance(score, BICCondGauss)
        assert score_c is score

    def test_get_scoring_method_by_name(self, small_df):
        data = small_df.astype("category")
        score, score_c = get_scoring_method("k2", data, use_cache=False)

        assert isinstance(score, K2)
        assert score_c is score

    def test_get_scoring_method_instance_passthrough(self, small_df):
        data = small_df.astype("category")
        score = K2(data)
        returned_score, score_c = get_scoring_method(score, data, use_cache=False)

        assert returned_score is score
        assert score_c is score

    def test_get_scoring_method_use_cache_is_noop(self, small_df):
        data = small_df.astype("category")
        score, score_c = get_scoring_method("k2", data, use_cache=True)

        assert isinstance(score, K2)
        assert score_c is score

    def test_get_scoring_method_unknown_score_error(self, small_df):
        with pytest.raises(ValueError, match=r"Unknown scoring method: 'not-a-score'"):
            get_scoring_method("not-a-score", small_df, use_cache=False)

    def test_get_scoring_method_none_without_data_error(self):
        with pytest.raises(
            ValueError, match=r"Cannot determine scoring method: both `scoring_method` and `data` are None."
        ):
            get_scoring_method(None, None, use_cache=False)

    def test_get_scoring_method_name_without_data_error(self):
        with pytest.raises(ValueError, match=r"Scoring method 'K2' requires data, but data is None."):
            get_scoring_method("k2", None, use_cache=False)

    def test_get_scoring_method_invalid_argument_error(self, small_df):
        with pytest.raises(ValueError, match=r"Invalid `scoring_method` argument: 123"):
            get_scoring_method(123, small_df, use_cache=False)
