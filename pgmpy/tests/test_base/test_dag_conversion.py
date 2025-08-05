#!/usr/bin/env python3
"""
Test for DAG to_lavaan and to_dagitty conversion methods.

This module tests the conversion functionality that allows pgmpy DAG objects
to be converted to lavaan and dagitty syntax representations for interoperability
with R packages.
"""

import pytest

from pgmpy.base import DAG


class TestDAGConversion:
    """Test for DAG to_lavaan and to_dagitty conversion methods"""

    def test_to_lavaan_simple_dag(self):
        """Test conversion of simple DAG to lavaan syntax"""
        dag = DAG([("X", "Y"), ("Z", "Y")])
        result = dag.to_lavaan()
        expected = "Y ~ X + Z"
        assert result == expected

    def test_to_lavaan_chain_dag(self):
        """Test conversion of chain DAG to lavaan syntax"""
        dag = DAG([("A", "B"), ("B", "C")])
        result = dag.to_lavaan()
        expected = "B ~ A\nC ~ B"
        assert result == expected

    def test_to_lavaan_complex_dag(self):
        """Test conversion of complex DAG to lavaan syntax"""
        dag = DAG([("A", "C"), ("B", "C"), ("C", "D"), ("A", "D")])
        result = dag.to_lavaan()
        expected = "C ~ A + B\nD ~ A + C"
        assert result == expected

    def test_to_lavaan_empty_dag(self):
        """Test conversion of empty DAG to lavaan syntax"""
        dag = DAG()
        result = dag.to_lavaan()
        assert result == ""

    def test_to_lavaan_isolated_nodes(self):
        """Test lavaan conversion with isolated nodes (no edges)"""
        dag = DAG()
        dag.add_nodes_from(["A", "B", "C"])
        result = dag.to_lavaan()
        assert result == ""  # No edges = no lavaan equations

    def test_to_lavaan_disconnected_components(self):
        """Test lavaan conversion with disconnected components"""
        dag = DAG([("A", "B"), ("C", "D")])
        result = dag.to_lavaan()
        expected = "B ~ A\nD ~ C"
        assert result == expected

    def test_to_dagitty_simple_dag(self):
        """Test conversion of simple DAG to dagitty syntax"""
        dag = DAG([("X", "Y"), ("Z", "Y")])
        result = dag.to_dagitty()
        expected = "dag {\nX -> Y\nZ -> Y\n}"
        assert result == expected

    def test_to_dagitty_chain_dag(self):
        """Test conversion of chain DAG to dagitty syntax"""
        dag = DAG([("A", "B"), ("B", "C")])
        result = dag.to_dagitty()
        expected = "dag {\nA -> B\nB -> C\n}"
        assert result == expected

    def test_to_dagitty_complex_dag(self):
        """Test conversion of complex DAG to dagitty syntax"""
        dag = DAG([("A", "C"), ("B", "C"), ("C", "D"), ("A", "D")])
        result = dag.to_dagitty()
        expected = "dag {\nA -> C\nA -> D\nB -> C\nC -> D\n}"
        assert result == expected

    def test_to_dagitty_empty_dag(self):
        """Test conversion of empty DAG to dagitty syntax"""
        dag = DAG()
        result = dag.to_dagitty()
        expected = "dag {\n}"
        assert result == expected

    def test_to_dagitty_isolated_nodes(self):
        """Test dagitty conversion with isolated nodes"""
        dag = DAG()
        dag.add_nodes_from(["A", "B", "C"])
        dag.add_edge("A", "B")
        result = dag.to_dagitty()
        expected = "dag {\nA -> B\nC\n}"
        assert result == expected

    def test_to_dagitty_only_isolated_nodes(self):
        """Test dagitty conversion with only isolated nodes"""
        dag = DAG()
        dag.add_nodes_from(["A", "B", "C"])
        result = dag.to_dagitty()
        expected = "dag {\nA\nB\nC\n}"
        assert result == expected

    def test_to_dagitty_disconnected_components(self):
        """Test dagitty conversion with disconnected components"""
        dag = DAG([("A", "B"), ("C", "D")])
        dag.add_node("E")  # Isolated node
        result = dag.to_dagitty()
        expected = "dag {\nA -> B\nC -> D\nE\n}"
        assert result == expected

    def test_numeric_node_names(self):
        """Test conversion with numeric node names"""
        dag = DAG([(1, 2), (3, 2)])

        lavaan_result = dag.to_lavaan()
        expected_lavaan = "2 ~ 1 + 3"
        assert lavaan_result == expected_lavaan

        dagitty_result = dag.to_dagitty()
        expected_dagitty = "dag {\n1 -> 2\n3 -> 2\n}"
        assert dagitty_result == expected_dagitty

    def test_mixed_node_name_types(self):
        """Test conversion with mixed node name types"""
        dag = DAG([("A", 1), (2, "B"), (1, 2)])

        lavaan_result = dag.to_lavaan()
        expected_lavaan = "1 ~ A\n2 ~ 1\nB ~ 2"
        assert lavaan_result == expected_lavaan

        dagitty_result = dag.to_dagitty()
        expected_dagitty = "dag {\n1 -> 2\n2 -> B\nA -> 1\n}"
        assert dagitty_result == expected_dagitty

    def test_special_character_node_names(self):
        """Test conversion with special characters in node names"""
        dag = DAG([("node_1", "node_2"), ("var-3", "node_2")])

        lavaan_result = dag.to_lavaan()
        expected_lavaan = "node_2 ~ node_1 + var-3"
        assert lavaan_result == expected_lavaan

        dagitty_result = dag.to_dagitty()
        expected_dagitty = "dag {\nnode_1 -> node_2\nvar-3 -> node_2\n}"
        assert dagitty_result == expected_dagitty

    def test_tuple_node_names(self):
        """Test conversion with tuple node names (hashable objects)"""
        dag = DAG([((1, 2), (3, 4)), ((5, 6), (3, 4))])

        lavaan_result = dag.to_lavaan()
        expected_lavaan = "(3, 4) ~ (1, 2) + (5, 6)"
        assert lavaan_result == expected_lavaan

        dagitty_result = dag.to_dagitty()
        expected_dagitty = "dag {\n(1, 2) -> (3, 4)\n(5, 6) -> (3, 4)\n}"
        assert dagitty_result == expected_dagitty

    def test_deterministic_output(self):
        """Test that output is deterministic (consistent ordering)"""
        # Create same DAG multiple times with different edge order
        dag1 = DAG([("Z", "Y"), ("X", "Y"), ("A", "B")])
        dag2 = DAG([("A", "B"), ("X", "Y"), ("Z", "Y")])

        assert dag1.to_lavaan() == dag2.to_lavaan()
        assert dag1.to_dagitty() == dag2.to_dagitty()

    def test_single_node_no_edges(self):
        """Test with single node and no edges"""
        dag = DAG()
        dag.add_node("A")

        lavaan_result = dag.to_lavaan()
        assert lavaan_result == ""

        dagitty_result = dag.to_dagitty()
        expected_dagitty = "dag {\nA\n}"
        assert dagitty_result == expected_dagitty

    def test_self_loops_prevention(self):
        """Test that self-loops are prevented (DAG constraint)"""
        # This should raise an error when creating the DAG
        with pytest.raises(ValueError, match="Cycles are not allowed"):
            DAG([("A", "A")])

    def test_large_dag(self):
        """Test with a larger DAG to ensure scalability"""
        edges = [(f"X{i}", f"X{i + 1}") for i in range(10)]
        edges.extend([("X0", "X5"), ("X2", "X7")])
        dag = DAG(edges)

        lavaan_result = dag.to_lavaan()
        dagitty_result = dag.to_dagitty()

        # Basic checks
        assert isinstance(lavaan_result, str)
        assert isinstance(dagitty_result, str)
        assert len(lavaan_result) > 0
        assert dagitty_result.startswith("dag {")
        assert dagitty_result.endswith("}")

    def test_round_trip_node_preservation(self):
        """Test that all nodes are preserved in some form"""
        # Test with disconnected components and isolated nodes
        dag = DAG([("A", "B"), ("C", "D")])
        dag.add_nodes_from(["E", "F"])  # Isolated nodes

        # For dagitty, all nodes should appear
        dagitty_result = dag.to_dagitty()
        for node in dag.nodes():
            assert str(node) in dagitty_result

        # For lavaan, only nodes with edges appear
        lavaan_result = dag.to_lavaan()
        assert "B" in lavaan_result  # Has parents
        assert "D" in lavaan_result  # Has parents
        # E and F are isolated, so they won't appear in lavaan

    def test_empty_string_vs_empty_structure(self):
        """Test difference between empty DAG and DAG with no edges"""
        # Completely empty DAG
        empty_dag = DAG()
        assert empty_dag.to_lavaan() == ""
        assert empty_dag.to_dagitty() == "dag {\n}"

        # DAG with nodes but no edges
        nodes_only_dag = DAG()
        nodes_only_dag.add_nodes_from(["A", "B"])
        assert nodes_only_dag.to_lavaan() == ""
        assert nodes_only_dag.to_dagitty() == "dag {\nA\nB\n}"

    def test_complex_multi_parent_structure(self):
        """Test with nodes having multiple parents"""
        dag = DAG(
            [
                ("A", "E"),
                ("B", "E"),
                ("C", "E"),
                ("D", "E"),  # E has 4 parents
                ("E", "F"),
                ("E", "G"),  # E has 2 children
                ("X", "Y"),  # Separate component
            ]
        )

        lavaan_result = dag.to_lavaan()
        assert "E ~ A + B + C + D" in lavaan_result
        assert "F ~ E" in lavaan_result
        assert "G ~ E" in lavaan_result
        assert "Y ~ X" in lavaan_result

        dagitty_result = dag.to_dagitty()
        assert "A -> E" in dagitty_result
        assert "B -> E" in dagitty_result
        assert "C -> E" in dagitty_result
        assert "D -> E" in dagitty_result
        assert "E -> F" in dagitty_result
        assert "E -> G" in dagitty_result
        assert "X -> Y" in dagitty_result

    def test_r_compatibility_note(self):
        """Test that node names with spaces/special chars work as documented"""
        # Node names with spaces (should work but may need quoting in R)
        dag_spaces = DAG([("node with spaces", "target")])
        lavaan_result = dag_spaces.to_lavaan()
        dagitty_result = dag_spaces.to_dagitty()

        assert "target ~ node with spaces" == lavaan_result
        assert "dag {\nnode with spaces -> target\n}" == dagitty_result

    def test_unicode_support(self):
        """Test that unicode characters in node names are supported"""
        dag_unicode = DAG([("α", "β"), ("γ", "β")])
        lavaan_result = dag_unicode.to_lavaan()
        dagitty_result = dag_unicode.to_dagitty()

        assert "β ~ α + γ" == lavaan_result
        assert "dag {\nα -> β\nγ -> β\n}" == dagitty_result
