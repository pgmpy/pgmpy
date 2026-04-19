import pandas as pd
import pytest

from pgmpy.estimators import (
    AIC,
    BIC,
    K2,
    AICCondGauss,
    AICGauss,
    BDeu,
    BDs,
    BICCondGauss,
    BICGauss,
    LogLikelihoodCondGauss,
    LogLikelihoodGauss,
)
from pgmpy.models import DiscreteBayesianNetwork

# Score values in the tests are compared to R package bnlearn


@pytest.fixture
def small_df():
    return pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]})


@pytest.fixture
def small_df_models():
    m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
    m2 = DiscreteBayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])
    return m1, m2


@pytest.fixture
def titanic_data():
    # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")
    return data[["Survived", "Sex", "Pclass"]]


@pytest.fixture
def bds_df():
    """Example taken from https://arxiv.org/pdf/1708.00689.pdf"""
    return pd.DataFrame(
        data={
            "X": [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            "Y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            "Z": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "W": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        }
    )


@pytest.fixture
def bds_models():
    m1 = DiscreteBayesianNetwork([("W", "X"), ("Z", "X")])
    m1.add_node("Y")
    m2 = DiscreteBayesianNetwork([("W", "X"), ("Z", "X"), ("Y", "X")])
    return m1, m2


@pytest.fixture
def aic_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")
    return AICGauss(data)


@pytest.fixture
def bic_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")
    return BICGauss(data)


@pytest.fixture
def bic_cond_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)
    return BICCondGauss(data)


@pytest.fixture
def aic_cond_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)
    return AICCondGauss(data)


@pytest.fixture
def loglik_cond_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)
    return LogLikelihoodCondGauss(data)


@pytest.fixture
def loglik_cond_gauss_manual_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)
    return LogLikelihoodCondGauss(data.iloc[:2, :])


@pytest.fixture
def loglik_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")
    return LogLikelihoodGauss(data)


@pytest.fixture
def gauss_models():
    m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
    m2 = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])
    return m1, m2


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


class TestBDs:
    def test_score(self, bds_df, bds_models):
        m1, m2 = bds_models
        scorer = BDs(bds_df, equivalent_sample_size=1)
        assert scorer.score(m1) == pytest.approx(-36.82311976667139)
        assert scorer.score(m2) == pytest.approx(-45.788991276221964)


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


class TestLogLikeGauss:
    def test_local_score_no_parents(self, loglik_gauss_score):
        # score(model2network("[A]"), df[c('A')], type='loglik-g') -> -119.7228
        assert loglik_gauss_score.local_score(variable="A", parents=[]) == pytest.approx(-119.7202, abs=1e-3)

        # score(model2network("[B]"), df[c('B')], type='loglik-g') -> -257.0067
        assert loglik_gauss_score.local_score(variable="B", parents=[]) == pytest.approx(-257.0042, abs=1e-3)

        # score(model2network("[C]"), df[c('C')], type='loglik-g')
        assert loglik_gauss_score.local_score(variable="C", parents=[]) == pytest.approx(-328.2361, abs=1e-3)

    def test_local_score_with_parents(self, loglik_gauss_score):
        # score(model2network("[A][B][C|A:B]"), df[c('A', 'B', 'C')], type='loglik-g') -> -455.1339
        assert loglik_gauss_score.local_score(variable="C", parents=["A", "B"]) == pytest.approx(-78.3815, abs=1e-3)

        # score(model2network("[A][B][C][D|A:B:C]"), df[c('A', 'B', 'C', 'D')], type='loglik-g') -> -732.2027
        assert loglik_gauss_score.local_score(variable="D", parents=["A", "B", "C"]) == pytest.approx(
            -27.1936, abs=1e-3
        )

    def test_score(self, loglik_gauss_score, gauss_models):
        m1, m2 = gauss_models
        assert loglik_gauss_score.score(m1) == pytest.approx(-455.1058, abs=1e-3)
        assert loglik_gauss_score.score(m2) == pytest.approx(-569.4505, abs=1e-3)


class TestAICGauss:
    def test_local_score_no_parents(self, aic_gauss_score):
        # score(model2network("[A]"), df_cont[c('A')], type='aic-g') -> -121.7228
        assert aic_gauss_score.local_score(variable="A", parents=[]) == pytest.approx(-121.7202, abs=1e-3)

        # score(model2network("[B]"), df_cont[c('B')], type='aic-g') -> -259.0067
        assert aic_gauss_score.local_score(variable="B", parents=[]) == pytest.approx(-259.0042, abs=1e-3)

        # score(model2network("[C]"), df_cont[c('C')], type='aic-g') -> -330.2386
        assert aic_gauss_score.local_score(variable="C", parents=[]) == pytest.approx(-330.2361, abs=1e-3)

    def test_local_score_with_parents(self, aic_gauss_score):
        # score(model2network("[A][B][C|A:B]"), df_cont[c('A', 'B', 'C')], type='aic-g') -> -463.1339
        assert aic_gauss_score.local_score(variable="C", parents=["A", "B"]) == pytest.approx(-82.3815, abs=1e-3)

        # score(model2network("[A][B][C][D|A:B:C]"), df_cont[c('A', 'B', 'C', 'D')], type='aic-g')
        assert aic_gauss_score.local_score(variable="D", parents=["A", "B", "C"]) == pytest.approx(-32.1936, abs=1e-3)

    def test_score(self, aic_gauss_score, gauss_models):
        m1, m2 = gauss_models
        assert aic_gauss_score.score(m1) == pytest.approx(-463.1059, abs=1e-3)
        assert aic_gauss_score.score(m2) == pytest.approx(-577.4505, abs=1e-3)


class TestBICGauss:
    def test_local_score_no_parents(self, bic_gauss_score):
        assert bic_gauss_score.local_score(variable="A", parents=[]) == pytest.approx(-124.3254, abs=1e-3)
        assert bic_gauss_score.local_score(variable="B", parents=[]) == pytest.approx(-261.6093, abs=1e-3)

    def test_local_score_with_parents(self, bic_gauss_score):
        assert bic_gauss_score.local_score(variable="C", parents=["A", "B"]) == pytest.approx(-87.5918, abs=1e-3)

    def test_score(self, bic_gauss_score, gauss_models):
        m1, m2 = gauss_models
        assert bic_gauss_score.score(m1) == pytest.approx(-473.5265, abs=1e-3)
        assert bic_gauss_score.score(m2) == pytest.approx(-587.8711, abs=1e-3)


class TestLogLikelihoodCondGauss:
    def test_score_manual_continuous(self, loglik_cond_gauss_manual_score):
        assert loglik_cond_gauss_manual_score.local_score(variable="A", parents=["B_cat"]) == pytest.approx(
            -1.8378, abs=1e-3
        )
        assert loglik_cond_gauss_manual_score.local_score(variable="A", parents=["B_cat", "B"]) == pytest.approx(
            -1.8379, abs=1e-3
        )

    def test_score_manual_categorical(self, loglik_cond_gauss_manual_score):
        assert loglik_cond_gauss_manual_score.local_score(variable="A_cat", parents=["B"]) == pytest.approx(
            2.9024, abs=1e-3
        )
        assert loglik_cond_gauss_manual_score.local_score(variable="A_cat", parents=["B_cat", "A"]) == pytest.approx(
            0, abs=1e-3
        )
        assert loglik_cond_gauss_manual_score.local_score(
            variable="A_cat", parents=["B", "B_cat", "C", "C_cat"]
        ) == pytest.approx(0, abs=1e-3)

    def test_score_bnlearn_no_parents(self, loglik_cond_gauss_score):
        # score(model2network("[A]"), d[c('A')], type='loglik-g') -> -119.7228
        assert loglik_cond_gauss_score.local_score(variable="A", parents=[]) == pytest.approx(-119.7228, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="B", parents=[]) == pytest.approx(-257.0067, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="B_cat", parents=[]) == pytest.approx(-81.6952, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="C", parents=[]) == pytest.approx(-328.2386, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="C_cat", parents=[]) == pytest.approx(-130.1208, abs=1e-3)
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=[]) == pytest.approx(-121.527, abs=1e-3)

    def test_score_bnlearn_continuous_parent(self, loglik_cond_gauss_score):
        # score(model2network("[B][A|B]"), d[c('A', 'B')], type='loglik-g') -> 376.5078
        assert loglik_cond_gauss_score.local_score(variable="A", parents=["B"]) == pytest.approx(-119.4935, abs=1e-3)

    def test_score_bnlearn_categorical_parent(self, loglik_cond_gauss_score):
        # score(model2network("[B_cat][A|B_cat]"), d[c('A', 'B_cat')], type='loglik-cg') -> 200.2201
        assert loglik_cond_gauss_score.local_score(variable="A", parents=["B_cat"]) == pytest.approx(
            -118.5250, abs=1e-3
        )
        # score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=["B_cat"]) == pytest.approx(
            -117.6219, abs=1e-3
        )

    def test_score_bnlearn_mixed_parents(self, loglik_cond_gauss_score):
        # score(model2network("[B][B_cat][A|B:B_cat]"), d[c('A', 'B', 'B_cat')], type='loglik-cg') -> 452.0991
        assert loglik_cond_gauss_score.local_score(variable="A", parents=["B_cat", "B"]) == pytest.approx(
            -113.2371, abs=1e-3
        )

    def test_score_bnlearn_many_parents(self, loglik_cond_gauss_score):
        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), type='loglik-cg') -> -Inf
        assert loglik_cond_gauss_score.local_score(variable="A", parents=["B_cat", "B", "C_cat", "C"]) == pytest.approx(
            19.1557, abs=1e-3
        )

    def test_score_bnlearn_continuous_to_categorical(self, loglik_cond_gauss_score):
        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=["B"]) == pytest.approx(
            -116.7104, abs=1e-3
        )
        assert loglik_cond_gauss_score.local_score(variable="A_cat", parents=["B_cat", "A"]) == pytest.approx(
            -6.1599, abs=1e-3
        )
        assert loglik_cond_gauss_score.local_score(
            variable="A_cat", parents=["B", "B_cat", "C", "C_cat"]
        ) == pytest.approx(41.9122, abs=1e-3)


class TestAICCondGauss:
    def test_score_bnlearn_no_parents(self, aic_cond_gauss_score):
        assert aic_cond_gauss_score.local_score(variable="B_cat", parents=[]) == pytest.approx(-83.6952, abs=1e-3)
        assert aic_cond_gauss_score.local_score(variable="B", parents=[]) == pytest.approx(-259.0067, abs=1e-3)
        assert aic_cond_gauss_score.local_score(variable="C", parents=[]) == pytest.approx(-330.2386, abs=1e-3)
        assert aic_cond_gauss_score.local_score(variable="C_cat", parents=[]) == pytest.approx(-134.1208, abs=1e-3)
        # score(model2network("[A_cat]"), d[c('A_cat')], type='loglik') -> -121.527
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=[]) == pytest.approx(-124.527, abs=1e-3)

    def test_score_bnlearn_categorical_parent(self, aic_cond_gauss_score):
        # score(model2network("[B_cat][A|B_cat]"), d[c('A', 'B_cat')], type='aic-cg') -> 208.2201
        assert aic_cond_gauss_score.local_score(variable="A", parents=["B_cat"]) == pytest.approx(-124.525, abs=1e-3)
        # score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=["B_cat"]) == pytest.approx(
            -126.6219, abs=1e-3
        )

    def test_score_bnlearn_mixed_parents(self, aic_cond_gauss_score):
        # score(model2network("[B][B_cat][A|B:B_cat]"), d[c('A', 'B', 'B_cat')], type='loglik-cg') -> 465.0991
        assert aic_cond_gauss_score.local_score(variable="A", parents=["B_cat", "B"]) == pytest.approx(
            -122.2372, abs=1e-3
        )

    def test_score_bnlearn_many_parents(self, aic_cond_gauss_score):
        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), type='loglik-cg') -> -Inf
        assert aic_cond_gauss_score.local_score(variable="A", parents=["B_cat", "B", "C_cat", "C"]) == pytest.approx(
            -40.8443, abs=1e-3
        )

    def test_score_bnlearn_continuous_to_categorical(self, aic_cond_gauss_score):
        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=["B"]) == pytest.approx(-125.7104, abs=1e-3)
        assert aic_cond_gauss_score.local_score(variable="A_cat", parents=["B_cat", "A"]) == pytest.approx(
            -33.1599, abs=1e-3
        )
        assert aic_cond_gauss_score.local_score(
            variable="A_cat", parents=["B", "B_cat", "C", "C_cat"]
        ) == pytest.approx(-138.0878, abs=1e-3)


class TestBICCondGauss:
    def test_score_bnlearn_no_parents(self, bic_cond_gauss_score):
        assert bic_cond_gauss_score.local_score(variable="B_cat", parents=[]) == pytest.approx(-86.3004, abs=1e-3)
        assert bic_cond_gauss_score.local_score(variable="B", parents=[]) == pytest.approx(-261.6119, abs=1e-3)
        assert bic_cond_gauss_score.local_score(variable="C", parents=[]) == pytest.approx(-332.8438, abs=1e-3)
        assert bic_cond_gauss_score.local_score(variable="C_cat", parents=[]) == pytest.approx(-139.3311, abs=1e-3)
        # score(model2network("[A_cat]"), d[c('A_cat')], type='loglik') -> -121.527
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=[]) == pytest.approx(-128.4347, abs=1e-3)

    def test_score_bnlearn_categorical_parent(self, bic_cond_gauss_score):
        # score(model2network("[B_cat][A|B_cat]"), d[c('A', 'B_cat')], type='bic-cg') -> 218.6408
        assert bic_cond_gauss_score.local_score(variable="A", parents=["B_cat"]) == pytest.approx(-132.3405, abs=1e-3)
        # score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=["B_cat"]) == pytest.approx(
            -138.3452, abs=1e-3
        )

    def test_score_bnlearn_mixed_parents(self, bic_cond_gauss_score):
        # score(model2network("[B][B_cat][A|B:B_cat]"), d[c('A', 'B', 'B_cat')], type='loglik-cg') -> 482.0327
        assert bic_cond_gauss_score.local_score(variable="A", parents=["B_cat", "B"]) == pytest.approx(
            -133.9605, abs=1e-3
        )

    def test_score_bnlearn_many_parents(self, bic_cond_gauss_score):
        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), type='loglik-cg') -> -Inf
        assert bic_cond_gauss_score.local_score(variable="A", parents=["B_cat", "B", "C_cat", "C"]) == pytest.approx(
            -118.9994, abs=1e-3
        )

    def test_score_bnlearn_continuous_to_categorical(self, bic_cond_gauss_score):
        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=["B"]) == pytest.approx(-137.4337, abs=1e-3)
        assert bic_cond_gauss_score.local_score(variable="A_cat", parents=["B_cat", "A"]) == pytest.approx(
            -68.3297, abs=1e-3
        )
        assert bic_cond_gauss_score.local_score(
            variable="A_cat", parents=["B", "B_cat", "C", "C_cat"]
        ) == pytest.approx(-372.5531, abs=1e-3)
