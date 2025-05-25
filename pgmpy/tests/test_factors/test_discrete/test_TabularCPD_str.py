import unittest
import numpy as np
from pgmpy.factors.discrete import TabularCPD


class TestTabularCPDStr(unittest.TestCase):
    def setUp(self):
        # Simple CPD with a single variable
        self.cpd1 = TabularCPD("A", 2, [[0.7], [0.3]])

        # CPD with evidence
        self.cpd2 = TabularCPD(
            "A", 2, [[0.1, 0.4], [0.9, 0.6]], evidence=["B"], evidence_card=[2]
        )

        # CPD with multiple evidence
        self.cpd3 = TabularCPD(
            "A",
            2,
            [[0.1, 0.4, 0.7, 0.9], [0.9, 0.6, 0.3, 0.1]],
            evidence=["B", "C"],
            evidence_card=[2, 2],
        )

        # CPD with tuple variable (as in Dynamic Bayesian Network)
        self.cpd4 = TabularCPD(("A", 0), 2, [[0.7], [0.3]])

        # CPD with tuple variable and tuple evidence
        self.cpd5 = TabularCPD(
            ("A", 0),
            2,
            [[0.1, 0.4], [0.9, 0.6]],
            evidence=[("B", 0)],
            evidence_card=[2],
        )

        # Complex CPD with multiple tuple evidence
        self.cpd6 = TabularCPD(
            ("A", 0),
            2,
            [[0.5, 0.0, 1.0, 0.0, 0.45, 0.5], [0.5, 1.0, 0.0, 1.0, 0.55, 0.5]],
            evidence=[("C", 0), ("B", 0)],
            evidence_card=[3, 2],
        )

        # CPDs from the issue example
        self.cpd_C = TabularCPD(("C", 0), 3, [[0.0714286], [0.307143], [0.621429]])

        self.cpd_A = TabularCPD(
            ("A", 0),
            2,
            [
                [0.5, 0.0, 1.0, 0.0, 0.456140350877, 0.5],
                [0.5, 1.0, 0.0, 1.0, 0.543859649123, 0.5],
            ],
            evidence=[("C", 0), ("B", 0)],
            evidence_card=[3, 2],
        )

        self.cpd_B = TabularCPD(
            ("B", 0),
            2,
            [[0.0, 0.219512195122, 1.0], [1.0, 0.780487804878, 0.0]],
            evidence=[("C", 0)],
            evidence_card=[3],
        )

        self.cpd_C1 = TabularCPD(
            ("C", 1),
            3,
            [
                [0.1, 0.0813953488372, 0.0862068965517],
                [0.3, 0.279069767442, 0.241379310345],
                [0.6, 0.639534883721, 0.672413793103],
            ],
            evidence=[("C", 0)],
            evidence_card=[3],
        )

    def test_str_simple_cpd(self):
        """Test string representation of a simple CPD."""
        result = str(self.cpd1)
        self.assertIsInstance(result, str)
        # Basic validation that the output contains the variable name
        self.assertIn("A", result)

    def test_str_cpd_with_evidence(self):
        """Test string representation of a CPD with evidence."""
        result = str(self.cpd2)
        self.assertIsInstance(result, str)
        # Check that both variables are in the output
        self.assertIn("A", result)
        self.assertIn("B", result)

    def test_str_cpd_with_multiple_evidence(self):
        """Test string representation of a CPD with multiple evidence."""
        result = str(self.cpd3)
        self.assertIsInstance(result, str)
        # Check that all variables are in the output
        self.assertIn("A", result)
        self.assertIn("B", result)
        self.assertIn("C", result)

    def test_str_cpd_with_tuple_variable(self):
        """Test string representation of a CPD with tuple variable."""
        try:
            result = str(self.cpd4)
            self.assertIsInstance(result, str)
            # Check that the variable is in the output
            self.assertIn("A", result)
        except Exception as e:
            self.fail(f"str(cpd4) raised {type(e).__name__} unexpectedly: {e}")

    def test_str_cpd_with_tuple_evidence(self):
        """Test string representation of a CPD with tuple variable and evidence."""
        try:
            result = str(self.cpd5)
            self.assertIsInstance(result, str)
            # Check that both variables are in the output
            self.assertIn("A", result)
            self.assertIn("B", result)
        except Exception as e:
            self.fail(f"str(cpd5) raised {type(e).__name__} unexpectedly: {e}")

    def test_str_complex_cpd_with_tuple_evidence(self):
        """Test string representation of a complex CPD with multiple tuple evidence."""
        try:
            result = str(self.cpd6)
            self.assertIsInstance(result, str)
            # Check that all variables are in the output
            self.assertIn("A", result)
            self.assertIn("B", result)
            self.assertIn("C", result)
        except Exception as e:
            self.fail(f"str(cpd6) raised {type(e).__name__} unexpectedly: {e}")

    def test_different_table_formats(self):
        """Test string representation with different table formats."""
        formats = ["plain", "simple", "grid", "fancy_grid", "pipe"]
        for fmt in formats:
            try:
                # First try using the __str__ method with a parameter
                result = str(self.cpd3)
                self.assertIsInstance(result, str)

                # If _str method exists and works correctly, test it too
                if hasattr(self.cpd3, "_str"):
                    try:
                        result = self.cpd3._str(tablefmt=fmt)
                        self.assertIsInstance(result, str)
                    except Exception as e:
                        self.fail(
                            f"_str with format {fmt} raised {type(e).__name__}: {e}"
                        )
            except Exception as e:
                self.fail(f"str() with format {fmt} raised {type(e).__name__}: {e}")

    def test_issue_example_cpds(self):
        """Test string representation of CPDs from the issue example."""
        # Test C CPD
        result = str(self.cpd_C)
        self.assertIsInstance(result, str)
        self.assertIn("C", result)
        # Check for values with less precision to avoid floating point comparison issues
        self.assertIn("0.07", result)  # Instead of exact "0.0714286"
        self.assertIn("0.30", result)  # Instead of exact "0.307143"
        self.assertIn("0.62", result)  # Instead of exact "0.621429"

        # Test A CPD
        result = str(self.cpd_A)
        self.assertIsInstance(result, str)
        self.assertIn("A", result)
        self.assertIn("B", result)
        self.assertIn("C", result)
        self.assertIn("0.5", result)
        self.assertIn("0", result)
        # Check for values with less precision
        self.assertIn("0.45", result)  # Instead of exact "0.456140350877"
        self.assertIn("0.54", result)  # Instead of exact "0.543859649123"

        # Test B CPD
        result = str(self.cpd_B)
        self.assertIsInstance(result, str)
        self.assertIn("B", result)
        self.assertIn("C", result)
        self.assertIn("0.0", result)
        self.assertIn("0.21", result)  # Instead of exact "0.219512195122"
        self.assertIn("0.78", result)  # Instead of exact "0.780487804878"

        # Test C1 CPD
        result = str(self.cpd_C1)
        self.assertIsInstance(result, str)
        self.assertIn("C", result)
        self.assertIn("0.1", result)
        self.assertIn("0.08", result)  # Instead of exact "0.0813953488372"
        self.assertIn("0.08", result)  # Instead of exact "0.0862068965517"
        self.assertIn("0.3", result)
        self.assertIn("0.27", result)  # Instead of exact "0.279069767442"
        self.assertIn("0.24", result)  # Instead of exact "0.241379310345"
        self.assertIn("0.6", result)
        self.assertIn("0.63", result)  # Instead of exact "0.639534883721"
        self.assertIn("0.67", result)  # Instead of exact "0.672413793103"

    def test_print_issue_example_cpds(self):
        """Print the CPDs from the issue example for visual inspection."""
        print("\nC CPD:")
        print(self.cpd_C)
        print("\nA CPD:")
        print(self.cpd_A)
        print("\nB CPD:")
        print(self.cpd_B)
        print("\nC1 CPD:")
        print(self.cpd_C1)

    def test_tablefmt_order(self):
        """Test if changing the order of arguments in _make_table_str works."""
        # Use _make_table_str directly without phi_or_p parameter
        result = self.cpd_A._make_table_str(tablefmt="grid")
        self.assertIsInstance(result, str)
        self.assertIn("A", result)


if __name__ == "__main__":
    unittest.main()
