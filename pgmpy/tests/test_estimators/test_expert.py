import os
import unittest

import networkx as nx
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

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

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_estimate(self):
        true_edges = [
            # Education-related paths
            ("Age", "Education"),
            ("Race", "Education"),
            ("NativeCountry", "Education"),
            # Income-related paths
            ("Education", "Income"),
            ("Occupation", "Income"),
            ("HoursPerWeek", "Income"),
            ("MaritalStatus", "Income"),
            # Occupation-related paths
            ("Age", "Occupation"),
            ("Education", "Occupation"),
            ("Sex", "Occupation"),
            ("Workclass", "Occupation"),
            # HoursPerWeek-related paths
            ("Age", "HoursPerWeek"),
            ("Workclass", "HoursPerWeek"),
            ("Occupation", "HoursPerWeek"),
            ("Education", "HoursPerWeek"),
            # Relationship and MaritalStatus paths
            ("Age", "MaritalStatus"),
            ("Sex", "MaritalStatus"),
            ("MaritalStatus", "Relationship"),
            ("Age", "Relationship"),
            ("Sex", "Relationship"),
            # Other reasonable connections
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

        # Use the expert estimator with our oracle orientation function
        estimated_dag = self.estimator.estimate(
            orientation_fn=oracle_orient,
            pval_threshold=0.05,
            effect_size_threshold=0.05,
            show_progress=True,
        )

        for u, v in estimated_dag.edges():
            self.assertTrue(true_dag.has_edge(u, v))

        self.assertTrue(nx.is_directed_acyclic_graph(estimated_dag))

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
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

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
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

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_estimate_with_custom_orient_fn(self):
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

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_estimate_with_orient_fn_kwargs(self):
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

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_combined_expert_knowledge(self):
        """Test combination of forbidden edges, required edges, and temporal order."""
        expert_knowledge = ExpertKnowledge(
            forbidden_edges=[("Age", "Income")],
            required_edges=[("Education", "Income")],
            temporal_order=[["Age", "Race"], ["Education"], ["Income", "HoursPerWeek"]],
        )

        # Run the algorithm
        dag = self.estimator.estimate(
            expert_knowledge=expert_knowledge,
            effect_size_threshold=0.0001,
            show_progress=False,
        )

        # Check forbidden edges
        assert ("Age", "Income") not in dag.edges()

        # Check temporal order
        for u, v in dag.edges():
            u_order = expert_knowledge.temporal_ordering[u]
            v_order = expert_knowledge.temporal_ordering[v]
            assert u_order <= v_order, f"Edge {u}->{v} violates temporal order"

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_edge_orientation_priority(self):
        """Test that edge orientation follows the correct priority order."""
        expert_knowledge = ExpertKnowledge(
            temporal_order=[["Age", "Race"], ["Education"], ["Income", "HoursPerWeek"]]
        )

        # Define orientations that should take precedence over temporal order
        orientations = {("Income", "Education")}  # Opposite of temporal order

        # Run the algorithm
        dag = self.estimator.estimate(
            expert_knowledge=expert_knowledge,
            orientations=orientations,
            effect_size_threshold=0.0001,
            show_progress=False,
        )

        # Check that specified orientations take precedence
        if ("Income", "Education") in dag.edges():
            assert ("Education", "Income") not in dag.edges()
