import numpy as np
import pandas as pd
import pytest

from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import DiscreteFactor


@pytest.fixture
def phi1():
    return DiscreteFactor(["x1", "x2", "x3"], [2, 2, 2], range(8))


@pytest.fixture
def phi2():
    return DiscreteFactor(["x4", "x5", "x6"], [2, 2, 2], range(8))


@pytest.fixture
def data_with_nans():
    return pd.DataFrame(
        data={
            "A": [0, np.nan, 1],
            "B": [0, 1, 0],
            "C": [1, 1, np.nan],
            "D": [np.nan, "Y", np.nan],
        }
    )


@pytest.fixture
def titanic_data():
    return pd.DataFrame.from_records(
        [
            [
                "35-49",
                "Academic-Degree",
                "Never-married",
                "White",
                "Male",
                "40",
                "no",
                "<=50K",
            ],
            [
                "50-65",
                "Academic-Degree",
                "Is-Married",
                "White",
                "Male",
                "<20",
                "no",
                "<=50K",
            ],
            ["35-49", "HS-grad", "Was-Married", "White", "Male", "40", "no", "<=50K"],
            [
                "50-65",
                "Non-HS-Grad",
                "Is-Married",
                "Non-White",
                "Male",
                "40",
                "no",
                "<=50K",
            ],
            [
                "20-34",
                "Academic-Degree",
                "Is-Married",
                "Non-White",
                "Female",
                "40",
                "yes",
                "<=50K",
            ],
        ],
        columns=[
            "Age",
            "Education",
            "MaritalStatus",
            "Race",
            "Sex",
            "HoursPerWeek",
            "Immigrant",
            "Income",
        ],
    )


class TestFactorDict:
    def test_class_init(self, phi1, phi2):
        factor_dict = FactorDict({tuple(i.scope()): i for i in [phi1, phi2]})
        assert {phi1, phi2} == factor_dict.get_factors()

    def test_factor_dict_addition_scalar(self, phi1):
        factor_dict = FactorDict({tuple(phi1.scope()): phi1})
        assert {phi1 + 2} == (factor_dict + 2).get_factors()

    def test_factor_dict_addition(self, phi1):
        factor_dict1 = FactorDict({tuple(phi1.scope()): phi1})
        factor_dict2 = FactorDict({tuple(phi1.scope()): phi1})
        assert {phi1 * 2} == (factor_dict1 + factor_dict2).get_factors()

    def test_factor_dict_multiplication(self, phi1):
        factor_dict = FactorDict({tuple(phi1.scope()): phi1})
        assert {phi1 * 2} == (factor_dict * 2).get_factors()

    def test_factor_dict_from_pandas_numeric(self):
        data = pd.DataFrame(
            data={"A": [0, 0, 1, 1], "B": [0, 1, 0, 1], "C": [1, 1, 0, 1]}
        )
        marginal = ("A", "B")
        factor_dict = FactorDict.from_dataframe(df=data, marginals=[marginal])
        factor = factor_dict[marginal]
        frequencies = data.value_counts(
            subset=list(marginal), sort=False, dropna=False
        ).values
        assert np.all(factor.values.flatten() == frequencies)

    def test_factor_dict_from_pandas_nans(self, data_with_nans):
        with pytest.raises(ValueError):
            FactorDict.from_dataframe(data_with_nans, ["A", "B"])

    def test_factor_dict_from_pandas_categorical(self):
        data = pd.DataFrame(
            data={
                "A": ["A", "B", "A", "B"],
                "B": ["A", "A", "A", "A"],
                "C": ["A", "A", "B", "B"],
            }
        )
        marginal = ("A", "C")
        factor_dict = FactorDict.from_dataframe(df=data, marginals=[marginal])
        factor = factor_dict[marginal]
        frequencies = data.value_counts(
            subset=list(marginal), sort=False, dropna=False
        ).values
        assert np.all(factor.values.flatten() == frequencies)

    def test_factor_dict_from_pandas_wrong_column(self):
        data = pd.DataFrame(
            data={"A": [0, 0, 1, 1], "B": [0, 1, 0, 1], "C": [1, 1, 0, 1]}
        )
        with pytest.raises(KeyError):
            FactorDict.from_dataframe(data, ["cheeseburger"])

    def test_factor_dict_from_pandas_titanic(self, titanic_data):
        marginal1 = ("Race", "Sex", "Income")
        marginal2 = ("Race", "Sex")
        marginal3 = ("Age", "HoursPerWeek")
        factor_dict = FactorDict.from_dataframe(
            df=titanic_data, marginals=[marginal1, marginal2, marginal3]
        )
        assert np.all(
            factor_dict[marginal1].values == np.array([[[1.0], [1.0]], [[0.0], [3.0]]])
        )
        assert np.all(
            factor_dict[marginal2].values == np.array([[1.0, 1.0], [0.0, 3.0]])
        )
        assert np.all(
            factor_dict[marginal3].values
            == np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
        )
