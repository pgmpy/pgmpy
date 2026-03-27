from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


def test_do_operator():
    model = DiscreteBayesianNetwork([("A", "B")])

    cpd_A = TabularCPD(variable="A", variable_card=2, values=[[0.6], [0.4]])

    cpd_B = TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.7, 0.2], [0.3, 0.8]],
        evidence=["A"],
        evidence_card=[2],
    )

    model.add_cpds(cpd_A, cpd_B)

    new_model = model.do(["B"])

    assert new_model.get_cpds("A") is not None
    assert new_model.get_cpds("B") is not None

    assert ("A", "B") not in new_model.edges()

    cpd_B_new = new_model.get_cpds("B")
    assert cpd_B_new.get_evidence() == []
