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


class CountingScore(BaseStructureScore):
    def __init__(self, data):
        self.call_count = 0
        super().__init__(data)

    def local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        self.call_count += 1
        parents = self._validate_parents(parents)
        return float(len(parents))


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
        score = get_scoring_method(None, data)

        assert isinstance(score, BIC)

    def test_get_scoring_method_default_continuous(self):
        data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")

        score = get_scoring_method(None, data)

        assert isinstance(score, BICGauss)

    def test_get_scoring_method_default_mixed(self):
        data = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)

        score = get_scoring_method(None, data)

        assert isinstance(score, BICCondGauss)

    def test_get_scoring_method_by_name(self, small_df):
        data = small_df.astype("category")
        score = get_scoring_method("k2", data)

        assert isinstance(score, K2)

    def test_get_scoring_method_instance_passthrough(self, small_df):
        data = small_df.astype("category")
        score = K2(data)
        returned_score = get_scoring_method(score, data)

        assert returned_score is score

    def test_get_scoring_method_returns_cached_score_instance(self, small_df):
        data = small_df.astype("category")
        score = get_scoring_method("k2", data)

        assert isinstance(score, K2)
        assert score.local_score("A", ()) == score.local_score("A", ())

    def test_base_structure_score_caches_local_score_calls(self, small_df):
        data = small_df.astype("category")
        score = CountingScore(data)

        assert score.local_score("A", ("B",)) == 1.0
        assert score.local_score("A", ("B",)) == 1.0
        assert score.call_count == 1

    def test_get_scoring_method_instance_passthrough_preserves_cached_score(self, small_df):
        data = small_df.astype("category")
        score = CountingScore(data)

        returned_score = get_scoring_method(score, data)
        assert returned_score is score

        assert returned_score.local_score("A", ("B",)) == 1.0
        assert returned_score.local_score("A", ("B",)) == 1.0
        assert score.call_count == 1

    def test_get_scoring_method_unknown_score_error(self, small_df):
        with pytest.raises(ValueError, match=r"Unknown scoring method: 'not-a-score'"):
            get_scoring_method("not-a-score", small_df)

    def test_get_scoring_method_none_without_data_error(self):
        with pytest.raises(
            ValueError, match=r"Cannot determine scoring method: both `scoring_method` and `data` are None."
        ):
            get_scoring_method(None, None)

    def test_get_scoring_method_name_without_data_error(self):
        with pytest.raises(ValueError, match=r"Scoring method 'K2' requires data, but data is None."):
            get_scoring_method("k2", None)

    def test_get_scoring_method_invalid_argument_error(self, small_df):
        with pytest.raises(ValueError, match=r"Invalid `scoring_method` argument: 123"):
            get_scoring_method(123, small_df)
