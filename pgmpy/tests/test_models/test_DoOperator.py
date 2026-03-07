import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


def test_do_reduces_child_cpd():
    model = DiscreteBayesianNetwork([("X", "Y")])


    cpd_x = TabularCPD(
        variable="X",
        variable_card=2,
        values=[[0.5], [0.5]]
    )

    cpd_y = TabularCPD(
        variable="Y",
        variable_card=2,
        values=[[0.8, 0.3],
                [0.2, 0.7]],
        evidence=["X"],
        evidence_card=[2]
    )

    model.add_cpds(cpd_x, cpd_y)

    intervened_model = model.do(["X"])


    new_cpd_y = intervened_model.get_cpds("Y")

    expected_values = np.array([0.8, 0.2])

    assert np.allclose(new_cpd_y.values, expected_values)
"""
NUmpy is used to compare the values of the new CPD for Y after intervention with the expected values. 
The test checks if the do operator correctly reduces the CPD of Y to reflect the intervention on X.
DiscreteBayesianNetwork is used to create a simple Bayesian network with two variables, X and Y, where X is a parent of Y.
TabularCPD is used to define the CPDs for X and Y. 
for x
P(X)
    X=0 ----> 0.5
    X=1  0.5
for y
P(Y | X)
    when x=0
        Y=0  ----> 0.8
        Y=1  ----> 0.2
    when x=1
        Y=0  ----> 0.3
        Y=1  ----> 0.7
The test then applies the do operator to intervene on X and checks if the resulting CPD for Y is as expected.
new CDP for Y should reflect the probabilities of Y given the intervention on X,
 which in this case should be [0.8, 0.2] since we are intervening on X and setting it to a specific value.
 Numpy's allclose function is used to check if the values of the new CPD for Y are close to the expected values, 
 allowing for some numerical precision issues.
 I tested this DoOperator functionality to ensure that the intervention correctly modifies the CPD of
   the child variable (Y) based on the parent variable (X) being intervened upon.
I used pytest to run this test function, and it should pass if the do operator is implemented correctly in the pgmpy library.
It has been succefully passed without any error.

I have made changes in __init__.py file (pgmpy/factors/discrete/__init__.py). I changed the order of imports to ensure that the
 DiscreteFactor and State are imported before TabularCPD(it was creating the circular import issue),
 as they are used in the TabularCPD class.

"""