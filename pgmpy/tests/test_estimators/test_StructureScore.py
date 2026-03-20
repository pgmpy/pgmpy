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
    RKHSCVLikelihood,
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


class TestRKHSCVLikelihood(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        n = 200

        # Linear chain: X1 -> X2
        X1_lin = rng.standard_normal(n)
        X2_lin = 0.8 * X1_lin + 0.2 * rng.standard_normal(n)
        cls.data_linear = pd.DataFrame({"X1": X1_lin, "X2": X2_lin})

        # Nonlinear chain: X1 -> X2 -> X3
        X1 = rng.standard_normal(n)
        X2 = np.sin(X1) + 0.2 * rng.standard_normal(n)
        X3 = 0.8 * X2 + 0.2 * rng.standard_normal(n)
        cls.data_nonlinear = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})

    def test_score_returns_float(self):
        scorer = RKHSCVLikelihood(self.data_linear)
        score = scorer.local_score("X2", ["X1"])
        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))
        self.assertFalse(np.isinf(score))

    def test_score_no_parents(self):
        scorer = RKHSCVLikelihood(self.data_linear)
        score = scorer.local_score("X1", [])
        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))

    def test_dependent_beats_independent(self):
        scorer = RKHSCVLikelihood(self.data_linear)
        score_with = scorer.local_score("X2", ["X1"])
        score_without = scorer.local_score("X2", [])
        self.assertGreater(score_with, score_without)

    def test_nonlinear_detection(self):
        scorer = RKHSCVLikelihood(self.data_nonlinear)
        score_with = scorer.local_score("X2", ["X1"])
        score_without = scorer.local_score("X2", [])
        self.assertGreater(score_with, score_without)

    def test_multiple_parents(self):
        scorer = RKHSCVLikelihood(self.data_nonlinear)
        score = scorer.local_score("X3", ["X1", "X2"])
        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))

    def test_string_key_registration(self):
        from pgmpy.estimators.StructureScore import get_scoring_method

        scorer, _ = get_scoring_method("rkhs-cv", self.data_linear, use_cache=False)
        self.assertIsInstance(scorer, RKHSCVLikelihood)

    def test_custom_hyperparameters(self):
        scorer = RKHSCVLikelihood(
            self.data_linear, n_folds=5, lambda_reg=0.1, gamma_noise=0.05
        )
        score = scorer.local_score("X2", ["X1"])
        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))

    def test_small_dataset(self):
        rng = np.random.default_rng(0)
        small_data = pd.DataFrame(
            {"A": rng.standard_normal(20), "B": rng.standard_normal(20)}
        )
        scorer = RKHSCVLikelihood(small_data, n_folds=5)
        score = scorer.local_score("B", ["A"])
        self.assertIsInstance(score, float)

    def test_kernel_flexibility(self):
        scorer = RKHSCVLikelihood(self.data_linear, kernel="laplacian")
        score = scorer.local_score("X2", ["X1"])
        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))

    def test_score_cache_compatible(self):
        from pgmpy.estimators.StructureScore import get_scoring_method

        _, scorer_cached = get_scoring_method(
            "rkhs-cv", self.data_linear, use_cache=True
        )
        score1 = scorer_cached.local_score("X2", ["X1"])
        score2 = scorer_cached.local_score("X2", ["X1"])
        self.assertEqual(score1, score2)

    def test_independent_variables_no_benefit(self):
        """Adding an independent variable as parent should not improve the score.

        For truly independent variables, CV likelihood penalizes the
        unnecessary complexity. The score with an irrelevant parent should
        be worse (lower) than without, because the model overfits noise
        in the training folds and generalizes poorly on the test folds.
        """
        rng = np.random.default_rng(123)
        n = 300
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)  # independent of A
        data = pd.DataFrame({"A": A, "B": B})
        scorer = RKHSCVLikelihood(data)
        score_no_parent = scorer.local_score("B", [])
        score_with_parent = scorer.local_score("B", ["A"])
        # CV penalizes overfitting: irrelevant parent should yield
        # equal or worse score
        self.assertGreaterEqual(score_no_parent, score_with_parent)

    def test_quadratic_nonlinearity(self):
        """Detect quadratic relationship: X2 = 0.8*(X1 + X1^2) + noise."""
        rng = np.random.default_rng(42)
        n = 300
        X1 = rng.standard_normal(n)
        X2 = 0.8 * (X1 + X1**2) + 0.2 * rng.standard_normal(n)
        data = pd.DataFrame({"X1": X1, "X2": X2})
        scorer = RKHSCVLikelihood(data)
        score_with = scorer.local_score("X2", ["X1"])
        score_without = scorer.local_score("X2", [])
        self.assertGreater(score_with, score_without)

    def test_nonlinear_chain_no_spurious_edge(self):
        """
        In a chain X1 -> X2 -> X3 with quadratic relationships,
        the RKHS score should:
        1. Prefer X2 as parent of X3 over X1 (direct vs indirect)
        2. Not benefit from adding X1 beyond X2 (conditional independence)

        This tests the scenario from the issue where BIC adds spurious X1->X3
        because partial correlation != 0 for nonlinear mechanisms.
        """
        rng = np.random.default_rng(42)
        n = 300
        X1 = rng.standard_normal(n)
        X2 = 0.8 * (X1 + X1**2) + 0.2 * rng.standard_normal(n)
        X3 = 0.8 * (X2 + X2**2) + 0.2 * rng.standard_normal(n)
        data = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
        scorer = RKHSCVLikelihood(data)

        score_x3_given_x2 = scorer.local_score("X3", ["X2"])
        score_x3_given_x1 = scorer.local_score("X3", ["X1"])
        score_x3_given_x1x2 = scorer.local_score("X3", ["X1", "X2"])

        # Direct parent X2 should beat indirect X1
        self.assertGreater(score_x3_given_x2, score_x3_given_x1)

        # Adding X1 beyond X2 should not help (X3 _|_ X1 | X2)
        # CV penalizes overfitting, so score(X3|{X1,X2}) <= score(X3|X2)
        self.assertGreaterEqual(score_x3_given_x2, score_x3_given_x1x2)

    def test_global_score_with_model(self):
        """Test that StructureScore.score(model) sums local scores correctly."""
        scorer = RKHSCVLikelihood(self.data_nonlinear)
        model = DiscreteBayesianNetwork([("X1", "X2"), ("X2", "X3")])
        total = scorer.score(model)
        expected = (
            scorer.local_score("X1", [])
            + scorer.local_score("X2", ["X1"])
            + scorer.local_score("X3", ["X2"])
        )
        self.assertAlmostEqual(total, expected, places=10)

    def test_works_with_hillclimb(self):
        """Integration test: HillClimbSearch with rkhs-cv finds the edge."""
        from pgmpy.estimators import HillClimbSearch

        hc = HillClimbSearch(self.data_linear)
        model = hc.estimate(scoring_method="rkhs-cv")
        # Should find the X1-X2 relationship (direction may vary)
        self.assertGreater(len(model.edges()), 0)

    def test_works_with_ges(self):
        """Integration test: GES with rkhs-cv finds the edge."""
        from pgmpy.estimators import GES

        ges = GES(self.data_linear)
        model = ges.estimate(scoring_method="rkhs-cv")
        # Should find the X1-X2 relationship
        self.assertGreater(len(model.edges()), 0)

    def test_deterministic(self):
        """Same data and params must produce identical scores."""
        scorer1 = RKHSCVLikelihood(self.data_linear)
        scorer2 = RKHSCVLikelihood(self.data_linear)
        s1 = scorer1.local_score("X2", ["X1"])
        s2 = scorer2.local_score("X2", ["X1"])
        self.assertEqual(s1, s2)
