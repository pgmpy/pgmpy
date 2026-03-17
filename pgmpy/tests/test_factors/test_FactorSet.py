import pytest

from pgmpy.factors import FactorSet
from pgmpy.factors.discrete import DiscreteFactor


@pytest.fixture
def phi1():
    return DiscreteFactor(["x1", "x2", "x3"], [2, 3, 2], range(12))


@pytest.fixture
def phi2():
    return DiscreteFactor(["x3", "x4", "x1"], [2, 2, 2], range(8))


@pytest.fixture
def phi3():
    return DiscreteFactor(["x5", "x6", "x7"], [2, 2, 2], range(8))


@pytest.fixture
def phi4():
    return DiscreteFactor(["x5", "x7", "x8"], [2, 2, 2], range(8))


class TestFactorSet:
    def test_class_init(self, phi1, phi2):
        factor_set = FactorSet(phi1, phi2)
        assert {phi1, phi2} == factor_set.get_factors()

    def test_factorset_add_remove_factors(self, phi1, phi2):
        factor_set = FactorSet()
        factor_set.add_factors(phi1, phi2)
        assert {phi1, phi2} == factor_set.get_factors()
        factor_set.remove_factors(phi2)
        assert {phi1} == factor_set.get_factors()

    def test_factorset_product(self, phi1, phi2, phi3, phi4):
        factor_set1 = FactorSet(phi1, phi2)
        factor_set2 = FactorSet(phi3, phi4)
        factor_set3 = factor_set2.product(factor_set1, inplace=False)
        assert {phi1, phi2, phi3, phi4} == factor_set3.factors

    def test_factorset_divide(self):
        phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 3, 2], range(1, 13))
        phi2 = DiscreteFactor(["x3", "x4", "x1"], [2, 2, 2], range(1, 9))
        phi3 = DiscreteFactor(["x5", "x6", "x7"], [2, 2, 2], range(1, 9))
        phi4 = DiscreteFactor(["x5", "x7", "x8"], [2, 2, 2], range(1, 9))
        factor_set1 = FactorSet(phi1, phi2)
        factor_set2 = FactorSet(phi3, phi4)
        factor_set3 = factor_set2.divide(factor_set1, inplace=False)
        assert {
            phi3,
            phi4,
            phi1.identity_factor() / phi1,
            phi2.identity_factor() / phi2,
        } == factor_set3.factors

    def test_factorset_marginalize_inplace(self, phi1, phi2, phi3, phi4):
        factor_set = FactorSet(phi1, phi2, phi3, phi4)
        factor_set.marginalize(["x1", "x5"], inplace=True)

        phi1_equiv = list(
            filter(lambda x: set(x.scope()) == {"x2", "x3"}, factor_set.factors)
        )[0]
        assert phi1.marginalize(["x1"], inplace=False) == phi1_equiv

        phi2_equiv = list(
            filter(lambda x: set(x.scope()) == {"x4", "x3"}, factor_set.factors)
        )[0]
        assert phi2.marginalize(["x1"], inplace=False) == phi2_equiv

        phi3_equiv = list(
            filter(lambda x: set(x.scope()) == {"x6", "x7"}, factor_set.factors)
        )[0]
        assert phi3.marginalize(["x5"], inplace=False) == phi3_equiv

        phi4_equiv = list(
            filter(lambda x: set(x.scope()) == {"x8", "x7"}, factor_set.factors)
        )[0]
        assert phi4.marginalize(["x5"], inplace=False) == phi4_equiv

    def test_factorset_marginalize_not_inplace(self, phi1, phi2, phi3, phi4):
        factor_set = FactorSet(phi1, phi2, phi3, phi4)
        new_factor_set = factor_set.marginalize(["x1", "x5"], inplace=False)

        phi1_equiv = list(
            filter(lambda x: set(x.scope()) == {"x2", "x3"}, new_factor_set.factors)
        )[0]
        assert phi1.marginalize(["x1"], inplace=False) == phi1_equiv

        phi2_equiv = list(
            filter(lambda x: set(x.scope()) == {"x4", "x3"}, new_factor_set.factors)
        )[0]
        assert phi2.marginalize(["x1"], inplace=False) == phi2_equiv

        phi3_equiv = list(
            filter(lambda x: set(x.scope()) == {"x6", "x7"}, new_factor_set.factors)
        )[0]
        assert phi3.marginalize(["x5"], inplace=False) == phi3_equiv

        phi4_equiv = list(
            filter(lambda x: set(x.scope()) == {"x8", "x7"}, new_factor_set.factors)
        )[0]
        assert phi4.marginalize(["x5"], inplace=False) == phi4_equiv
