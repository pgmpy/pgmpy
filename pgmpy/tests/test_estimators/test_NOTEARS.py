import unittest

import networkx as nx
import numpy as np
import numpy.testing as np_test
import pandas as pd
import torch

from pgmpy import config
from pgmpy.estimators import NOTEARS, ExpertKnowledge
from pgmpy.metrics import SHD
from pgmpy.models import LinearGaussianBayesianNetwork as LGBN
from pgmpy.utils import get_example_model


class TestNoTEARS(unittest.TestCase):
    def setUp(self):
        self.rand_model = LGBN.get_random(
            n_nodes=5, node_names=["A", "B", "C", "D", "E"], loc=1, scale=0.2, seed=42
        )
        self.rand_data = self.rand_model.simulate(int(1e3), seed=42)
        self.est_rand = NOTEARS(self.rand_data)

        self.ecoli_model = get_example_model("ecoli70")
        self.ecoli_data = self.ecoli_model.simulate(int(1e3), seed=42)
        self.est_ecoli = NOTEARS(self.ecoli_data)

    def test_constraint_gradient(self):
        # Contains cycle - A->B->E->A
        W_est = np.array(
            [
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype="float64",
        )
        if config.get_backend() == "torch":
            W_est = torch.from_numpy(W_est)

        acyclic_penalty, acyclic_jac = self.est_rand._constraint_grad(W_est)
        self.assertNotEqual(float(acyclic_penalty), 0)
        self.assertAlmostEqual(float(acyclic_penalty), 0.5042, places=3)
        np_test.assert_array_almost_equal(
            acyclic_jac[acyclic_jac > 0], [1.0167, 1.0167, 1.0167], decimal=3
        )

        # True model
        W_est = nx.adjacency_matrix(nx.DiGraph(self.rand_model.edges())).todense()
        if config.get_backend() == "torch":
            W_est = torch.from_numpy(W_est).to(torch.float64)
            # W_est = W_est.to('float64')

        acyclic_penalty, acyclic_jac = self.est_rand._constraint_grad(W_est)
        self.assertAlmostEqual(float(acyclic_penalty), 0.0, places=3)
        self.assertTrue((acyclic_jac == 0).all())

    def test_expert_knowledge_loss(self):
        forbidden_edges = [("A", "B"), ("D", "C"), ("A", "E")]
        required_edges = [("C", "E"), ("A", "C")]
        nodes = self.rand_data.columns.to_list()
        forbidden_mask = np.zeros((5, 5))
        for u, v in forbidden_edges:
            i = nodes.index(u)
            j = nodes.index(v)
            forbidden_mask[i][j] = 1

        fixed_mask = np.zeros((5, 5))
        for u, v in required_edges:
            i = nodes.index(u)
            j = nodes.index(v)
            fixed_mask[i][j] = 1

        # Satisfies all the constraints
        W_est = np.array(
            [
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        if config.get_backend() == "torch":
            W_est = torch.from_numpy(W_est)
            fixed_mask = torch.from_numpy(fixed_mask)
            forbidden_mask = torch.from_numpy(forbidden_mask)

        fixed_penalty, fixed_jac = self.est_rand._fixed_penalty_gradient(
            W_est, fixed_mask, 10, 0.3
        )
        forbidden_penalty, forbidden_jac = self.est_rand._forbidden_penalty_gradient(
            W_est, forbidden_mask
        )

        self.assertEqual(float(fixed_penalty), 0.0)
        self.assertEqual(float(forbidden_penalty), 0.0)
        self.assertTrue(fixed_jac.all() == np.zeros((5, 5)).all())
        self.assertTrue(forbidden_jac.all() == np.zeros((5, 5)).all())

        # Satisfies forbidden edges but not required edges
        W_est = np.array(
            [
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        if config.get_backend() == "torch":
            W_est = torch.from_numpy(W_est)

        fixed_penalty, fixed_jac = self.est_rand._fixed_penalty_gradient(
            W_est, fixed_mask, 10, 0.3
        )
        forbidden_penalty, forbidden_jac = self.est_rand._forbidden_penalty_gradient(
            W_est, forbidden_mask
        )

        # TODO: Need to be fixed
        self.assertNotEqual(float(fixed_penalty), 0.0)
        self.assertEqual(float(forbidden_penalty), 0.0)
        self.assertTrue(fixed_jac.all() == np.zeros((5, 5)).all())
        self.assertTrue(forbidden_jac.all() == np.zeros((5, 5)).all())

        # Satisfies required edges but not forbidden edges
        W_est = np.array(
            [
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        if config.get_backend() == "torch":
            W_est = torch.from_numpy(W_est)

        fixed_penalty, fixed_jac = self.est_rand._fixed_penalty_gradient(
            W_est, fixed_mask, 10, 0.3
        )
        forbidden_penalty, forbidden_jac = self.est_rand._forbidden_penalty_gradient(
            W_est, forbidden_mask
        )

        # TODO: Need to be fixed
        self.assertEqual(float(fixed_penalty), 0.0)
        self.assertNotEqual(float(forbidden_penalty), 0.0)
        self.assertTrue(fixed_jac.all() == np.zeros((5, 5)).all())
        self.assertTrue(forbidden_jac.all() == np.zeros((5, 5)).all())

    def test_estimate_rand(self):
        est_dag = self.est_rand.estimate(
            lambda1=0.2, w_threshold=0.3, show_progress=False
        )
        self.assertEqual(set(est_dag.edges()), set(self.rand_model.edges()))

        """
        dag_rand_logistic = self.est_rand.estimate(
            lambda1=0.01, loss_type="logistic", show_progress=False
        )
        self.assertSetEqual(
            set(dag_rand_logistic.edges()), set([("A", "D"), ("A", "C"), ("B", "C")])
        )"""

    def test_estimate_expert(self):
        expert_knowledge = ExpertKnowledge(forbidden_edges=[("D", "C"), ("E", "B")])
        est_dag = self.est_rand.estimate(
            lambda1=0.01, expert_knowledge=expert_knowledge, show_progress=False
        )
        self.assertNotIn(set([("D", "C"), ("E", "B")]), est_dag.edges())

        expert_knowledge = ExpertKnowledge(fixed_edges=[("A", "B"), ("C", "D")])
        est_dag = self.est_rand.estimate(
            lambda1=0.01, expert_knowledge=expert_knowledge, show_progress=False
        )
        self.assertIn(
            set([("A", "B"), ("C", "D")]), est_dag.edges()
        )  # fixed penalty failing

        expert_knowledge = ExpertKnowledge(
            forbidden_edges=[("D", "C"), ("E", "B")],
            fixed_edges=[("A", "B"), ("C", "D")],
        )
        est_dag = self.est_rand.estimate(
            lambda1=0.01, expert_knowledge=expert_knowledge, show_progress=False
        )
        self.assertNotIn(set([("D", "C"), ("E", "B")]), est_dag.edges())
        self.assertIn(set([("A", "B"), ("C", "D")]), est_dag.edges())

    def test_estimate_ecoli(self):
        est_dag = self.est_ecoli.estimate(
            lambda1=0.8, w_threshold=0.2, show_progress=False
        )
        self.assertEqual(SHD(self.ecoli_model, est_dag), 56)

    def tearDown(self):
        del self.rand_model
        del self.rand_data
        del self.est_rand

        del self.ecoli_model
        del self.ecoli_data
        del self.est_ecoli


class TestNoTEARSTorch(TestNoTEARS):
    def setUp(self):
        config.set_backend("torch")
        super().setUp()

    def tearDown(self):
        super().tearDown()
        config.set_backend("numpy")
