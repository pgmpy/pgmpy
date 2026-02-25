import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import precision_recall

# -----------------------------------------------------------------------
# GROUP 1: Happy path — correct results on well-formed inputs (7 tests)
# -----------------------------------------------------------------------


def test_identical_dags_perfect_score():
    """Identical graphs → all metrics = 1.0"""
    dag = DAG([("X", "Y"), ("Y", "Z")])
    result = precision_recall(dag, dag)
    assert result["skeleton"]["precision"] == 1.0
    assert result["skeleton"]["recall"] == 1.0
    assert result["skeleton"]["f1"] == 1.0
    assert result["directed"]["f1"] == 1.0


def test_reversed_single_edge():
    """One edge reversed: skeleton perfect, directed drops, spurious v-structure"""
    # true: X→Y→Z   est: X→Y←Z
    # Skeleton: {X-Y, Y-Z} in both → perfect
    # Directed: X→Y correct (1 TP), Y→Z vs Z→Y (1 FP, 1 FN) → p=r=0.5
    # V-structure: est has X→Y←Z, true has none → TP=0, FP=1 → precision=0
    true = DAG([("X", "Y"), ("Y", "Z")])
    est = DAG([("X", "Y"), ("Z", "Y")])
    result = precision_recall(true, est)
    assert result["skeleton"]["f1"] == 1.0
    assert result["directed"]["precision"] == pytest.approx(0.5)
    assert result["directed"]["recall"] == pytest.approx(0.5)
    assert result["directed"]["f1"] == pytest.approx(0.5)
    assert result["v_structure"]["precision"] == 0.0
    assert result["v_structure"]["recall"] == 0.0


def test_missing_edge_in_estimated():
    """Est misses an edge → skeleton recall drops, precision stays 1"""
    true = DAG([("X", "Y"), ("Y", "Z")])
    est = DAG([("X", "Y")])
    est.add_node("Z")
    result = precision_recall(true, est)
    assert result["skeleton"]["precision"] == 1.0
    assert result["skeleton"]["recall"] == pytest.approx(0.5)
    assert result["skeleton"]["f1"] == pytest.approx(2 / 3)


def test_extra_edge_in_estimated():
    """Est has a spurious edge → skeleton precision drops, recall stays 1"""
    true = DAG([("X", "Y")])
    true.add_node("Z")
    est = DAG([("X", "Y"), ("Y", "Z")])
    result = precision_recall(true, est)
    assert result["skeleton"]["precision"] == pytest.approx(0.5)
    assert result["skeleton"]["recall"] == 1.0
    assert result["skeleton"]["f1"] == pytest.approx(2 / 3)


def test_v_structure_recovered_correctly():
    """V-structure in both → v_structure f1 = 1.0"""
    # X→Z←Y is a v-structure (X and Y not adjacent)
    true = DAG([("X", "Z"), ("Y", "Z")])
    est = DAG([("X", "Z"), ("Y", "Z")])
    result = precision_recall(true, est)
    assert result["v_structure"]["precision"] == 1.0
    assert result["v_structure"]["recall"] == 1.0
    assert result["v_structure"]["f1"] == 1.0


def test_v_structure_missed_by_estimated():
    """V-structure in true but not est → recall=0"""
    # true: X→Z←Y   est: X→Z→Y (chain, no v-structure)
    true = DAG([("X", "Z"), ("Y", "Z")])
    est = DAG([("X", "Z"), ("Z", "Y")])
    result = precision_recall(true, est)
    assert result["v_structure"]["recall"] == 0.0


def test_v_structure_false_positive():
    """V-structure in est but not true → precision=0"""
    # true: X→Z→Y   est: X→Z←Y (spurious v-structure)
    true = DAG([("X", "Z"), ("Z", "Y")])
    est = DAG([("X", "Z"), ("Y", "Z")])
    result = precision_recall(true, est)
    assert result["v_structure"]["precision"] == 0.0


# -----------------------------------------------------------------------
# GROUP 2: Zero-division guards — every branch where denominator = 0 (4 tests)
# These are the branches Codecov will flag if untested.
# -----------------------------------------------------------------------


def test_empty_estimated_no_division_error():
    """Est has no edges → TP+FP=0 → precision=0.0, not ZeroDivisionError"""
    true = DAG([("X", "Y"), ("Y", "Z")])
    est = DAG()
    est.add_nodes_from(["X", "Y", "Z"])
    result = precision_recall(true, est)
    assert result["skeleton"]["precision"] == 0.0
    assert result["directed"]["precision"] == 0.0
    assert result["v_structure"]["precision"] == 0.0


def test_empty_true_no_division_error():
    """True has no edges → TP+FN=0 → recall=0.0, not ZeroDivisionError"""
    true = DAG()
    true.add_nodes_from(["X", "Y", "Z"])
    est = DAG([("X", "Y"), ("Y", "Z")])
    result = precision_recall(true, est)
    assert result["skeleton"]["recall"] == 0.0
    assert result["directed"]["recall"] == 0.0


def test_both_empty_graphs_no_error():
    """Both graphs have no edges → all metrics = 0.0, no errors"""
    true = DAG()
    true.add_nodes_from(["X", "Y", "Z"])
    est = DAG()
    est.add_nodes_from(["X", "Y", "Z"])
    result = precision_recall(true, est)
    for metric in ["skeleton", "directed", "v_structure"]:
        assert result[metric]["precision"] == 0.0
        assert result[metric]["recall"] == 0.0
        assert result[metric]["f1"] == 0.0


def test_no_v_structures_in_either_graph_no_error():
    """Chain graphs have no v-structures → v_structure=0.0, no ZeroDivisionError"""
    # X→Y→Z: Y is not a collider, so no v-structures
    true = DAG([("X", "Y"), ("Y", "Z")])
    est = DAG([("X", "Y"), ("Y", "Z")])
    result = precision_recall(true, est)
    assert result["skeleton"]["f1"] == 1.0
    assert result["v_structure"]["f1"] == 0.0  # both empty → 0.0, not error


# -----------------------------------------------------------------------
# GROUP 3: Input validation branches (2 tests)
# -----------------------------------------------------------------------


def test_mismatched_node_sets_raises_value_error():
    """Different node sets → ValueError with informative message"""
    true = DAG([("X", "Y")])
    est = DAG([("A", "B")])
    with pytest.raises(ValueError):
        precision_recall(true, est)


def test_invalid_type_raises_type_error():
    """Non-DAG/PDAG input → TypeError"""
    with pytest.raises(TypeError):
        precision_recall("not_a_dag", DAG([("X", "Y")]))


# -----------------------------------------------------------------------
# GROUP 4: PDAG input — undirected edges (1 test)
# -----------------------------------------------------------------------


def test_pdag_estimated_skeleton_counts_undirected():
    """Undirected edge in PDAG est contributes to skeleton TP but not directed TP"""
    true = DAG([("X", "Y"), ("Y", "Z")])
    # PDAG: X-Y undirected, Y→Z directed
    est = PDAG(directed_ebunch=[("Y", "Z")], undirected_ebunch=[("X", "Y")])
    result = precision_recall(true, est)
    # Skeleton: X-Y counts regardless of direction → recall = 1.0
    assert result["skeleton"]["recall"] == 1.0
    # Directed: only Y→Z matches → f1 < 1.0 (X→Y not counted as directed match)
    assert result["directed"]["f1"] < 1.0


def test_pdag_true_skeleton_counts_undirected():
    """Undirected edge in PDAG true contributes to skeleton computations"""
    # PDAG: X-Y undirected, Y→Z directed
    true = PDAG(directed_ebunch=[("Y", "Z")], undirected_ebunch=[("X", "Y")])
    est = DAG([("X", "Y"), ("Y", "Z")])
    result = precision_recall(true, est)
    # Skeleton: X-Y matches undirected X-Y -> precision = 1.0, recall = 1.0
    assert result["skeleton"]["precision"] == 1.0
    assert result["skeleton"]["recall"] == 1.0
    # Directed: X->Y est does not match X-Y true directed -> precision drops
    assert result["directed"]["precision"] < 1.0


def test_pdag_v_structure():
    """Test v-structure extraction from a PDAG"""
    # X->Z<-Y with X-Y having no edge (directed or undirected)
    pdag = PDAG(directed_ebunch=[("X", "Z"), ("Y", "Z")])
    pdag.add_nodes_from(["X", "Y", "Z"])
    # It should correctly identify X->Z<-Y as a v-structure
    result = precision_recall(pdag, pdag)
    assert result["v_structure"]["f1"] == 1.0

    # Missing v-structure: X->Z, Y->Z, but X-Y is undirected edge
    pdag_no_vstruct = PDAG(
        directed_ebunch=[("X", "Z"), ("Y", "Z")], undirected_ebunch=[("X", "Y")]
    )
    result2 = precision_recall(pdag, pdag_no_vstruct)
    assert result2["v_structure"]["recall"] == 0.0


# -----------------------------------------------------------------------
# GROUP 5: Larger graph sanity check (1 test)
# -----------------------------------------------------------------------


def test_larger_graph_values_reasonable():
    """5-node graph: all metrics in [0,1], F1 = harmonic mean of P and R"""
    true = DAG([("A", "B"), ("B", "C"), ("A", "C"), ("C", "D"), ("D", "E")])
    est = DAG([("A", "B"), ("B", "C"), ("A", "D"), ("C", "D"), ("E", "D")])
    result = precision_recall(true, est)
    for metric in ["skeleton", "directed", "v_structure"]:
        p = result[metric]["precision"]
        r = result[metric]["recall"]
        f = result[metric]["f1"]
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0
        assert 0.0 <= f <= 1.0
        if p > 0 and r > 0:
            assert f == pytest.approx(2 * p * r / (p + r), abs=1e-5)
