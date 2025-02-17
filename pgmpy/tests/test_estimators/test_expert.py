import os
import unittest

import pandas as pd
import pytest

from pgmpy.estimators import ExpertInLoop


class TestExpertInLoop(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/adult_proc.csv", index_col=0
        )
        df.Age = pd.Categorical(
            df.Age,
            categories=["<21", "21-30", "31-40", "41-50", "51-60", "61-70", ">70"],
            ordered=True,
        )
        df.Education = pd.Categorical(
            df.Education,
            categories=[
                "Preschool",
                "1st-4th",
                "5th-6th",
                "7th-8th",
                "9th",
                "10th",
                "11th",
                "12th",
                "HS-grad",
                "Some-college",
                "Assoc-voc",
                "Assoc-acdm",
                "Bachelors",
                "Prof-school",
                "Masters",
                "Doctorate",
            ],
            ordered=True,
        )
        df.HoursPerWeek = pd.Categorical(
            df.HoursPerWeek, categories=["<=20", "21-30", "31-40", ">40"], ordered=True
        )
        df.Workclass = pd.Categorical(df.Workclass, ordered=False)
        df.MaritalStatus = pd.Categorical(df.MaritalStatus, ordered=False)
        df.Occupation = pd.Categorical(df.Occupation, ordered=False)
        df.Relationship = pd.Categorical(df.Relationship, ordered=False)
        df.Race = pd.Categorical(df.Race, ordered=False)
        df.Sex = pd.Categorical(df.Sex, ordered=False)
        df.NativeCountry = pd.Categorical(df.NativeCountry, ordered=False)
        df.Income = pd.Categorical(df.Income, ordered=False)

        self.estimator = ExpertInLoop(data=df)
        self.descriptions = {
            "Age": "The age of a person",
            "Workclass": "The workplace where the person is employed such as Private industry, or self employed",
            "Education": "The highest level of education the person has finished",
            "MaritalStatus": "The marital status of the person",
            "Occupation": "The kind of job the person does. For example, sales, craft repair, clerical",
            "Relationship": "The relationship status of the person",
            "Race": "The ethnicity of the person",
            "Sex": "The sex or gender of the person",
            "HoursPerWeek": "The number of hours per week the person works",
            "NativeCountry": "The native country of the person",
            "Income": "The income i.e. amount of money the person makes",
        }
        self.estimator_small = ExpertInLoop(
            data=df[["Age", "Education", "Race", "Sex", "Income"]]
        )
        self.orientations_small = {
            ("Education", "Income"),
            ("Race", "Education"),
            ("Age", "Education"),
        }

    @pytest.mark.skipif(
        "GEMINI_API_KEY" not in os.environ, reason="Gemini API key is not set"
    )
    def test_estimate(self):
        dag = self.estimator.estimate(variable_descriptions=self.descriptions)
        # expected_edges = {('MaritalStatus', 'Relationship'), ('Age', 'Occupation'), ('NativeCountry', 'MaritalStatus'), ('Sex', 'Occupation'), ('Occupation', 'Income'), ('HoursPerWeek', 'Income'), ('NativeCountry', 'Education'), ('Age', 'HoursPerWeek'), ('Workclass', 'Occupation'), ('Education', 'Income'), ('Age', 'Workclass'), ('MaritalStatus', 'Income'), ('Workclass', 'HoursPerWeek'), ('NativeCountry', 'HoursPerWeek'), ('Education', 'Occupation'), ('Occupation', 'HoursPerWeek'), ('Age', 'Relationship'), ('Race', 'NativeCountry'), ('Sex', 'Relationship'), ('Education', 'HoursPerWeek'), ('Race', 'Education'), ('Workclass', 'Relationship'), ('MaritalStatus', 'HoursPerWeek'), ('Age', 'MaritalStatus'), ('Sex', 'MaritalStatus'), ('Relationship', 'HoursPerWeek'), ('Age', 'Education'), ('Workclass', 'MaritalStatus')}
        # self.assertEqual(expected_edges, set(dag.edges()))

    def test_estimate_with_orientations(self):
        orientations = self.orientations_small
        dag = self.estimator_small.estimate(
            variable_descriptions=self.descriptions,
            use_llm=False,
            orientations=orientations,
            pval_threshold=0.1,
            effect_size_threshold=0.1,
        )
        self.assertEqual(orientations, set(dag.edges()))
        self.assertEqual(self.estimator_small.orientations_llm, set([]))

    def test_estimate_with_cache_no_llm_calls(self):
        orientations = self.orientations_small
        self.estimator_small.orientations_llm = orientations
        dag = self.estimator_small.estimate(
            variable_descriptions=self.descriptions,
            use_cache=True,
            use_llm=True,
            orientations=orientations,
            pval_threshold=0.1,
            effect_size_threshold=0.1,
        )
        self.assertEqual(orientations, set(dag.edges()))
        self.assertEqual(self.estimator_small.orientations_llm, orientations)

    @pytest.mark.skipif(
        "GEMINI_API_KEY" not in os.environ, reason="Gemini API key is not set"
    )
    def test_estimate_with_cache_and_llm_calls(self):
        orientations = self.orientations_small
        dag = self.estimator_small.estimate(
            variable_descriptions=self.descriptions,
            use_cache=True,
            use_llm=True,
            orientations=orientations,
            pval_threshold=0.1,
            effect_size_threshold=0.1,
        )
        self.assertEqual(orientations, set(dag.edges()))
        self.assertEqual(self.estimator_small.orientations_llm, orientations)
