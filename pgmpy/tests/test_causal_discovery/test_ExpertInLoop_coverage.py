"""
Targeted tests to cover specifically missing lines in ExpertInLoop coverage report.
"""

import logging
import sys
from unittest.mock import patch, MagicMock

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertInLoop
from pgmpy.estimators import ExpertKnowledge
from pgmpy.ci_tests._base import _BaseCITest

def simple_orient(var1, var2, **kwargs):
    return (var1, var2) if var1 < var2 else (var2, var1)

class StrongCI(_BaseCITest):
    def __init__(self, data):
        self.data = data
        super().__init__()

    def run_test(self, X, Y, Z):
        return (0.5, 0.001)

def test_get_edge_orientation_orientations_ctor():
    est = ExpertInLoop(orientations={("A", "B")})
    assert est._get_edge_orientation("A", "B") == ("A", "B")
    assert est._get_edge_orientation("B", "A") == ("A", "B")
    ek = ExpertKnowledge(temporal_order=[["B"], ["A"]]) # B before A
    est = ExpertInLoop(expert_knowledge=ek, orientations={("A", "B")})
    # orientations says A->B, but temporal says B must come before A (lower order score)
    # wait, ExpertKnowledge.temporal_ordering returns a dict {var: order_index}
    # temporal_order=[["B"], ["A"]] means B: 0, A: 1.
    # So B before A. If orientations says A->B, and B < A, then A->B is actually VALID.
    # Wait, the code says: if to[res[0]] > to[res[1]]: res = (res[1], res[0])
    # to['A'] = 1, to['B'] = 0. res = ('A', 'B'). res[0]='A', res[1]='B'.
    # 1 > 0 is True, so it swaps to ('B', 'A').
    assert est._get_edge_orientation("A", "B") == ("B", "A")

def test_get_edge_orientation_temporal_tie():
    ek = ExpertKnowledge(temporal_order=[["A", "B"]])
    est = ExpertInLoop(expert_knowledge=ek)
    assert est._get_edge_orientation("A", "B") is None

def test_fit_merge_orientations_list():
    ek = ExpertKnowledge(orientations=[("A", "B")])
    est = ExpertInLoop(expert_knowledge=ek, orientations={("C", "D")})
    data = pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [1, 2], "D": [1, 2]})
    est.fit(data)
    assert set(est.expert_knowledge_.orientations) == {("A", "B"), ("C", "D")}

def test_fit_edge_removal():
    np.random.seed(42)
    data = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100)})
    
    est = ExpertInLoop(orientation_fn=simple_orient, pval_threshold=0.05, effect_size_threshold=0.05)
    est.fit(data)
    mock_effects = pd.DataFrame([
        ["A", "B", [], True, 0.01, 0.9] # Edge present but weak
    ], columns=["u", "v", "z", "edge_present", "effect", "p_val"])
    
    with patch.object(est, "_test_all", return_value=mock_effects):
        def side_effect(*args, **kwargs):
            if est.n_iter_ == 1:
                return pd.DataFrame([["A", "B", [], False, 0.8, 0.001]], 
                                  columns=["u", "v", "z", "edge_present", "effect", "p_val"])
            else:
                return pd.DataFrame([["A", "B", [], True, 0.01, 0.9]], 
                                  columns=["u", "v", "z", "edge_present", "effect", "p_val"])

        est = ExpertInLoop(orientation_fn=simple_orient, max_iter=2)
        with patch.object(est, "_test_all", side_effect=side_effect):
            est.fit(data)
    assert ("A", "B") not in est.causal_graph_.edges()

def test_fit_only_removals_iteration():
    np.random.seed(42)
    data = pd.DataFrame({"A": [1, 2], "B": [1, 2]})
    
    def side_effect(*args, **kwargs):
        if est.n_iter_ == 1:
            return pd.DataFrame([["A", "B", [], False, 0.8, 0.001]], 
                              columns=["u", "v", "z", "edge_present", "effect", "p_val"])
        elif est.n_iter_ == 2:
            return pd.DataFrame([["A", "B", [], True, 0.01, 0.9]], 
                              columns=["u", "v", "z", "edge_present", "effect", "p_val"])
        else:
            return pd.DataFrame([], columns=["u", "v", "z", "edge_present", "effect", "p_val"])

    est = ExpertInLoop(orientation_fn=simple_orient, max_iter=3)
    with patch.object(est, "_test_all", side_effect=side_effect):
        est.fit(data)
    assert est.n_iter_ == 3

def test_fit_cycle_rejection_logger(caplog):
    np.random.seed(0)
    data = pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [1, 2]})
    
    est = ExpertInLoop(orientation_fn=lambda u, v: ("C", "A"), 
                      show_progress=True, max_iter=10)
    
    def side_effect(*args, **kwargs):
        dag = kwargs.get('dag')
        if not dag.has_edge("A", "B"):
            return pd.DataFrame([["A", "B", [], False, 0.9, 0.0]], columns=["u", "v", "z", "edge_present", "effect", "p_val"])
        if not dag.has_edge("B", "C"):
            return pd.DataFrame([["B", "C", [], False, 0.9, 0.0]], columns=["u", "v", "z", "edge_present", "effect", "p_val"])
        if not dag.has_edge("C", "A"):
            return pd.DataFrame([["C", "A", [], False, 0.9, 0.0]], columns=["u", "v", "z", "edge_present", "effect", "p_val"])
        return pd.DataFrame([], columns=["u", "v", "z", "edge_present", "effect", "p_val"])

    eiul_module = sys.modules["pgmpy.causal_discovery.ExpertInLoop"]
    with patch.object(eiul_module, "get_ci_test", return_value=StrongCI(data)):
        def mock_orient(u, v):
            if set([u, v]) == {"A", "B"}: return ("A", "B")
            if set([u, v]) == {"B", "C"}: return ("B", "C")
            if set([u, v]) == {"C", "A"}: return ("C", "A")
            return None
        
        with patch.object(est, "_get_edge_orientation", side_effect=mock_orient):
            with caplog.at_level(logging.INFO, logger="pgmpy"):
                est.fit(data)
                
    assert "Cycle-breaking subroutine suggested removing the new edge" in caplog.text
    assert ("C", "A") not in est.causal_graph_.edges()
