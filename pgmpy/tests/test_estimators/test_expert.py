import os
import unittest

import pandas as pd
import pytest

from pgmpy.estimators import ExpertInLoop


@pytest.mark.skip("Temporarily skipping ExpertInLoop tests")
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
        # Check either attribute depending on which one exists
        orientations_cache = getattr(
            self.estimator_small,
            "orientation_cache",
            getattr(self.estimator_small, "orientations_llm", set([])),
        )
        self.assertEqual(orientations_cache, set([]))

    def test_estimate_with_cache_no_llm_calls(self):
        orientations = self.orientations_small
        # Set the appropriate attribute based on which one exists in the implementation
        if hasattr(self.estimator_small, "orientation_cache"):
            self.estimator_small.orientation_cache = orientations
        else:
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
        # Check either attribute depending on which one exists
        orientations_cache = getattr(
            self.estimator_small,
            "orientation_cache",
            getattr(self.estimator_small, "orientations_llm", set([])),
        )
        self.assertEqual(orientations_cache, orientations)

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
        # Check either attribute depending on which one exists
        orientations_cache = getattr(
            self.estimator_small,
            "orientation_cache",
            getattr(self.estimator_small, "orientations_llm", set([])),
        )
        self.assertEqual(orientations_cache, orientations)

    def test_estimate_with_custom_orientation_function(self):
        def custom_orient(var1, var2, **kwargs):
            # Always orient edges from alphabetically first to second
            if var1 < var2:
                return (var1, var2)
            else:
                return (var2, var1)

        dag = self.estimator_small.estimate(
            orientation_fn=custom_orient,
            pval_threshold=0.1,
            effect_size_threshold=0.1,
        )

        # Check that all edges are oriented from alphabetically lower to higher
        for edge in dag.edges():
            self.assertTrue(edge[0] < edge[1])

        # Check that orientations were cached
        self.assertTrue(len(self.estimator_small.orientation_cache) > 0)
        for edge in self.estimator_small.orientation_cache:
            self.assertTrue(edge[0] < edge[1])

    def test_estimate_with_invalid_orientation_function(self):
        def invalid_orient(var1, var2, **kwargs):
            # Return an invalid orientation (not a tuple of the right vars)
            return ("InvalidVar", var2)

        with self.assertRaises(ValueError):
            self.estimator_small.estimate(
                orientation_fn=invalid_orient,
                pval_threshold=0.1,
                effect_size_threshold=0.1,
            )

    def test_estimate_with_orientation_fn_kwargs_1(self):
        def orient_with_kwargs(var1, var2, **kwargs):
            # Use a keyword argument to determine orientation
            if kwargs.get("reverse_alphabetical", False):
                if var1 > var2:
                    return (var1, var2)
                else:
                    return (var2, var1)
            else:
                if var1 < var2:
                    return (var1, var2)
                else:
                    return (var2, var1)

        # Test with reverse_alphabetical=True
        dag_reverse = self.estimator_small.estimate(
            orientation_fn=orient_with_kwargs,
            reverse_alphabetical=True,
            pval_threshold=0.1,
            effect_size_threshold=0.1,
        )

        # Check that all edges are oriented from alphabetically higher to lower
        for edge in dag_reverse.edges():
            self.assertTrue(edge[0] > edge[1])

    def test_estimate_with_orientation_fn_kwargs_2(self):
        def orient_with_kwargs(var1, var2, **kwargs):
            # Use a keyword argument to determine orientation
            if kwargs.get("reverse_alphabetical", False):
                if var1 > var2:
                    return (var1, var2)
                else:
                    return (var2, var1)
            else:
                if var1 < var2:
                    return (var1, var2)
                else:
                    return (var2, var1)

        dag_normal = self.estimator_small.estimate(
            orientation_fn=orient_with_kwargs,
            pval_threshold=0.1,
            effect_size_threshold=0.1,
        )

        # Check that all edges are oriented from alphabetically lower to higher
        for edge in dag_normal.edges():
            self.assertTrue(edge[0] < edge[1])
