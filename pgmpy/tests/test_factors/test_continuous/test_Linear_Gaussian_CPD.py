import numpy as np
import pytest

from pgmpy.factors.continuous import LinearGaussianCPD


def test_class_init():
    beta = np.array([7, 13])
    std = np.array([[4, 3], [3, 6]])

    cpd1 = LinearGaussianCPD("Y", beta=beta, std=std, evidence=["X1", "X2"])
    assert cpd1.variable == "Y"
    assert cpd1.evidence == ["X1", "X2"]


def test_str():
    cpd1 = LinearGaussianCPD("x", [0.23], 0.56)
    cpd2 = LinearGaussianCPD("y", [0.67, 1, 4.56, 8], 2, ["x1", "x2", "x3"])
    assert cpd1.__str__() == "P(x) = N(0.23; 0.56)"
    assert (
        cpd2.__str__() == "P(y | x1, x2, x3) = N(1.0*x1 + 4.56*x2 + 8.0*x3 + 0.67; 2)"
    )


def test_get_random():
    cpd_random = LinearGaussianCPD.get_random("x", ["x1", "x2", "x3"], 0.23, 0.56)
    assert "P(x | x1, x2, x3) = N(" in cpd_random.__str__()


def test_variable_hashable():
    with pytest.raises(ValueError, match="argument must be hashable"):
        LinearGaussianCPD(variable=["X"], beta=[0], std=1)
