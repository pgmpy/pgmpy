import unittest

import numpy as np
import pandas as pd

from pgmpy.estimators import GES, PC, ExpertKnowledge, HillClimbSearch, MmhcEstimator


class TestRootNodeAlgorithms(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset for testing
        self.data = pd.DataFrame(
            np.random.randint(0, 2, size=(100, 4)), columns=["A", "B", "C", "D"]
        )
        self.data["A"] = self.data["B"] + self.data["C"]
        for col in self.data.columns:
            self.data[col] = self.data[col].astype("category")

    def test_pc_root_node(self):
        """Test PC algorithm with a root node."""
        expert_knowledge = ExpertKnowledge(root_nodes=["D"])
        est = PC(self.data)
        dag = est.estimate(
            variant="stable",
            expert_knowledge=expert_knowledge,
            significance_level=0.05,
        )
        self.assertEqual(len(list(dag.predecessors("D"))), 0)

    def test_hill_climb_root_node(self):
        """Test HillClimbSearch algorithm with a root node."""
        expert_knowledge = ExpertKnowledge(root_nodes=["D"])
        est = HillClimbSearch(self.data)
        dag = est.estimate(scoring_method="k2", expert_knowledge=expert_knowledge)
        self.assertEqual(len(list(dag.predecessors("D"))), 0)

    def test_ges_root_node(self):
        """Test GES algorithm with a root node."""
        expert_knowledge = ExpertKnowledge(root_nodes=["D"])
        est = GES(self.data)
        dag = est.estimate(scoring_method="k2", expert_knowledge=expert_knowledge)
        self.assertEqual(len(list(dag.predecessors("D"))), 0)

    def test_mmhc_root_node(self):
        """Test MmhcEstimator with a root node."""
        expert_knowledge = ExpertKnowledge(root_nodes=["D"])
        est = MmhcEstimator(self.data)
        dag = est.estimate(expert_knowledge=expert_knowledge)
        self.assertEqual(len(list(dag.predecessors("D"))), 0)

    def test_pc_multiple_root_nodes(self):
        """Test PC algorithm with multiple root nodes."""
        expert_knowledge = ExpertKnowledge(root_nodes=["C", "D"])
        est = PC(self.data)
        dag = est.estimate(
            variant="stable",
            expert_knowledge=expert_knowledge,
            significance_level=0.05,
        )
        self.assertEqual(len(list(dag.predecessors("C"))), 0)
        self.assertEqual(len(list(dag.predecessors("D"))), 0)

    def test_hill_climb_multiple_root_nodes(self):
        """Test HillClimbSearch algorithm with multiple root nodes."""
        expert_knowledge = ExpertKnowledge(root_nodes=["C", "D"])
        est = HillClimbSearch(self.data)
        dag = est.estimate(scoring_method="k2", expert_knowledge=expert_knowledge)
        self.assertEqual(len(list(dag.predecessors("C"))), 0)
        self.assertEqual(len(list(dag.predecessors("D"))), 0)

    def test_ges_multiple_root_nodes(self):
        """Test GES algorithm with multiple root nodes."""
        expert_knowledge = ExpertKnowledge(root_nodes=["C", "D"])
        est = GES(self.data)
        dag = est.estimate(scoring_method="k2", expert_knowledge=expert_knowledge)
        self.assertEqual(len(list(dag.predecessors("C"))), 0)
        self.assertEqual(len(list(dag.predecessors("D"))), 0)


if __name__ == "__main__":
    unittest.main()
