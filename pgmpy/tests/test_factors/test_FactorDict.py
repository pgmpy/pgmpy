import unittest

import numpy as np
import pandas as pd

from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import DiscreteFactor


class TestFactorDict(unittest.TestCase):
    def setUp(self):
        self.phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 2, 2], range(8))
        self.phi2 = DiscreteFactor(["x4", "x5", "x6"], [2, 2, 2], range(8))
        self.data1 = pd.DataFrame(
            data={"A": [0, 0, 1, 1], "B": [0, 1, 0, 1], "C": [1, 1, 0, 1]}
        )
        self.data2 = pd.DataFrame(
            data={
                "A": [0, np.nan, 1],
                "B": [0, 1, 0],
                "C": [1, 1, np.nan],
                "D": [np.nan, "Y", np.nan],
            }
        )
        self.data3 = pd.DataFrame(
            data={
                "A": ["A", "B", "A", "B"],
                "B": ["A", "A", "A", "A"],
                "C": ["A", "A", "B", "B"],
            }
        )
        self.titanic_data = pd.DataFrame.from_records(
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
                [
                    "35-49",
                    "HS-grad",
                    "Was-Married",
                    "White",
                    "Male",
                    "40",
                    "no",
                    "<=50K",
                ],
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

    def test_class_init(self):
        phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 2, 2], range(8))
        phi2 = DiscreteFactor(["x4", "x5", "x6"], [2, 2, 2], range(8))
        factor_dict = FactorDict({tuple(i.scope()): i for i in [phi1, phi2]})
        self.assertEqual({self.phi1, self.phi2}, factor_dict.get_factors())

    def test_factor_dict_addition_scalar(self):
        phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 2, 2], range(8))
        factor_dict1 = FactorDict({tuple(phi1.scope()): phi1})
        self.assertEqual({self.phi1 + 2}, (factor_dict1 + 2).get_factors())

    def test_factor_dict_addition(self):
        phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 2, 2], range(8))
        factor_dict1 = FactorDict({tuple(phi1.scope()): phi1})
        factor_dict2 = FactorDict({tuple(phi1.scope()): phi1})
        self.assertEqual({self.phi1 * 2}, (factor_dict1 + factor_dict2).get_factors())

    def test_factor_dict_multiplication(self):
        phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 2, 2], range(8))
        factor_dict1 = FactorDict({tuple(phi1.scope()): phi1})
        self.assertEqual({self.phi1 * 2}, (factor_dict1 * 2).get_factors())

    def test_factor_dict_from_pandas_numeric(self):
        marginal = ("A", "B")
        factor_dict = FactorDict.from_dataframe(df=self.data1, marginals=[marginal])
        factor = factor_dict[marginal]
        frequencies = self.data1.value_counts(
            subset=list(marginal), sort=False, dropna=False
        ).values
        self.assertTrue(np.all(factor.values.flatten() == frequencies))

    def test_factor_dict_from_pandas_nans(self):
        self.assertRaises(ValueError, FactorDict.from_dataframe, self.data2, ["A", "B"])

    def test_factor_dict_from_pandas_categorical(self):
        marginal = ("A", "C")
        factor_dict = FactorDict.from_dataframe(df=self.data3, marginals=[marginal])
        factor = factor_dict[marginal]
        frequencies = self.data3.value_counts(
            subset=list(marginal), sort=False, dropna=False
        ).values
        self.assertTrue(np.all(factor.values.flatten() == frequencies))

    def test_factor_dict_from_pandas_wrong_column(self):
        self.assertRaises(
            KeyError, FactorDict.from_dataframe, self.data1, ["cheeseburger"]
        )

    def test_factor_dict_from_pandas_titanic(self):
        marginal1 = ("Race", "Sex", "Income")
        race_sex_income = np.array([[[1.0], [1.0]], [[0.0], [3.0]]])
        marginal2 = ("Race", "Sex")
        marginal3 = ("Age", "HoursPerWeek")
        age_hoursperweek = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
        race_sex = np.array([[1.0, 1.0], [0.0, 3.0]])
        factor_dict = FactorDict.from_dataframe(
            df=self.titanic_data, marginals=[marginal1, marginal2, marginal3]
        )
        self.assertTrue(np.all(factor_dict[marginal1].values == race_sex_income))
        self.assertTrue(np.all(factor_dict[marginal2].values == race_sex))
        self.assertTrue(np.all(factor_dict[marginal3].values == age_hoursperweek))
