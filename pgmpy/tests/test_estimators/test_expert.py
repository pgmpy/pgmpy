import os
import unittest

import networkx as nx
import pandas as pd
import pytest

from pgmpy.estimators import ExpertInLoop, ExpertKnowledge


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
        # ExpertKnowledge for testing
        self.expert_knowledge = ExpertKnowledge(
            forbidden_edges=[("Income", "Education")],
            required_edges=[("Age", "Education")],
            temporal_order=["Age", "Race", "Education", "Sex", "Income"],
        )
        self.estimator_expert = ExpertInLoop(
            data=df[["Age", "Education", "Race", "Sex", "Income"]],
            expert_knowledge=self.expert_knowledge,
        )

    def test_estimate(self):
        true_edges = [
            ("Age", "Education"),
            ("Race", "Education"),
            ("NativeCountry", "Education"),
            ("Education", "Income"),
            ("Occupation", "Income"),
            ("HoursPerWeek", "Income"),
            ("MaritalStatus", "Income"),
            ("Age", "Occupation"),
            ("Education", "Occupation"),
            ("Sex", "Occupation"),
            ("Workclass", "Occupation"),
            ("Age", "HoursPerWeek"),
            ("Workclass", "HoursPerWeek"),
            ("Occupation", "HoursPerWeek"),
            ("Education", "HoursPerWeek"),
            ("Age", "MaritalStatus"),
            ("Sex", "MaritalStatus"),
            ("MaritalStatus", "Relationship"),
            ("Age", "Relationship"),
            ("Sex", "Relationship"),
            ("Race", "NativeCountry"),
            ("Workclass", "MaritalStatus"),
            ("Workclass", "Relationship"),
        ]

        true_dag = nx.DiGraph(true_edges)
        true_dag.add_nodes_from(self.estimator.data.columns)

        def oracle_orient(var1, var2, **kwargs):
            """Orientation function that knows the 'true' structure."""
            if true_dag.has_edge(var1, var2):
                return (var1, var2)
            elif true_dag.has_edge(var2, var1):
                return (var2, var1)
            else:
                return None

        estimated_dag = self.estimator.estimate(
            orientation_fn=oracle_orient,
            pval_threshold=0.05,
            effect_size_threshold=0.05,
            show_progress=True,
        )

        for u, v in estimated_dag.edges():
            self.assertTrue(true_dag.has_edge(u, v))

        self.assertTrue(nx.is_directed_acyclic_graph(estimated_dag))

    def test_estimate_with_orientations(self):
        orientations = self.orientations_small
        dag = self.estimator_small.estimate(
            pval_threshold=0.1,
            effect_size_threshold=0.1,
            orientations=orientations,
        )
        self.assertEqual(orientations, set(dag.edges()))
        orientations_cache = getattr(self.estimator_small, "orientation_cache", set([]))
        self.assertEqual(orientations_cache, set([]))

    def test_estimate_with_cache(self):
        self.estimator_small.orientation_cache = self.orientations_small

        dag = self.estimator_small.estimate(
            use_cache=True,
            pval_threshold=0.1,
            effect_size_threshold=0.1,
        )
        self.assertEqual(self.orientations_small, set(dag.edges()))
        orientations_cache = getattr(self.estimator_small, "orientation_cache", set([]))
        self.assertEqual(orientations_cache, self.orientations_small)

    def test_estimate_with_custom_orient_fn(self):
        def custom_orient(var1, var2, **kwargs):
            if var1 < var2:
                return (var1, var2)
            else:
                return (var2, var1)

        dag = self.estimator_small.estimate(
            orientation_fn=custom_orient,
            pval_threshold=0.1,
            effect_size_threshold=0.1,
        )

        for edge in dag.edges():
            self.assertTrue(edge[0] < edge[1])

        self.assertTrue(len(self.estimator_small.orientation_cache) > 0)
        for edge in self.estimator_small.orientation_cache:
            self.assertTrue(edge[0] < edge[1])

    def test_estimate_with_orient_fn_kwargs(self):
        def orient_with_kwargs(var1, var2, **kwargs):
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

        dag_reverse = self.estimator_small.estimate(
            orientation_fn=orient_with_kwargs,
            reverse_alphabetical=True,
            pval_threshold=0.1,
            effect_size_threshold=0.1,
        )

        for edge in dag_reverse.edges():
            self.assertTrue(edge[0] > edge[1])

    def test_forbidden_edges(self):
        """Test that forbidden edges are not added to the DAG."""
        dag = self.estimator_expert.estimate(
            pval_threshold=0.1,
            effect_size_threshold=0.1,
            orientation_fn=lambda x, y, **kwargs: (x, y) if x < y else (y, x),
        )
        self.assertNotIn(("Income", "Education"), dag.edges())
        self.assertIn(("Income", "Education"), self.estimator_expert.blacklisted_edges)

    def test_required_edges(self):
        """Test that required edges are included in the initial DAG."""
        self.assertIn(("Age", "Education"), self.estimator_expert.dag.edges())
        # Note: Required edges may be removed during pruning, so we don't test final DAG

    def test_temporal_order(self):
        """Test that temporal order resolves edge orientations."""
        dag = self.estimator_expert.estimate(
            pval_threshold=0.1,
            effect_size_threshold=0.1,
            orientation_fn=lambda x, y, **kwargs: None,  # Force temporal order to dominate
        )
        # Since Age precedes Education in temporal_order, expect Age -> Education
        for edge in dag.edges():
            if "Age" in edge and "Education" in edge:
                self.assertEqual(edge, ("Age", "Education"))
            # Since Race precedes Sex, expect Race -> Sex if edge exists
            if "Race" in edge and "Sex" in edge:
                self.assertEqual(edge, ("Race", "Sex"))
