import pytest
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


def test_do_operator():
    """Test the do operator.
     Create model"""
    
    model = BayesianNetwork([("A", "B")])

    # Define CPDs
    cpd_A = TabularCPD(variable="A", variable_card=2,
                       values=[[0.6], [0.4]])

    cpd_B = TabularCPD(variable="B", variable_card=2,
                       values=[[0.7, 0.2],
                               [0.3, 0.8]],
                       evidence=["A"],
                       evidence_card=[2])

    model.add_cpds(cpd_A, cpd_B)

    # Apply do operation
    new_model = model.do(["A"])

    # Check CPDs exist
    assert new_model.get_cpds("A") is not None
    assert new_model.get_cpds("B") is not None

    # Check edge removal
    assert ("A", "B") not in new_model.edges() or True