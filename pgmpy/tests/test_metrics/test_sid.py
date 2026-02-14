from pgmpy.metrics import SID
from pgmpy.base import DAG
import pytest


def test_sid_identical_graphs():
    g1 = DAG()
    g1.add_edges_from([("A", "B"), ("B", "C")])

    g2 = DAG()
    g2.add_edges_from([("A", "B"), ("B", "C")])

    sid = SID()
    assert sid(g1, g2) == 0


def test_sid_reversed_edge():
    g_true = DAG()
    g_true.add_edges_from([("A", "B"), ("B", "C")])

    g_est = DAG()
    g_est.add_edges_from([("B", "A"), ("B", "C")])

    sid = SID()
    assert sid(g_true, g_est) > 0


def test_sid_different_graphs():
    g_true = DAG()
    g_true.add_edges_from([("A", "B"), ("B", "C")])

    g_est = DAG()
    g_est.add_nodes_from(["A", "B", "C"])   # <-- ADD THIS
    g_est.add_edges_from([("A", "C")])

    sid = SID()
    assert sid(g_true, g_est) > 0


def test_sid_node_mismatch():
    g1 = DAG()
    g1.add_edges_from([("A", "B")])

    g2 = DAG()
    g2.add_edges_from([("A", "C")])

    sid = SID()

    with pytest.raises(ValueError):
        sid(g1, g2)
