import unittest

from pgmpy.factors import FactorSet
from pgmpy.factors.discrete import DiscreteFactor


class TestFactorSet(unittest.TestCase):
    def setUp(self):
        self.phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 3, 2], range(12))
        self.phi2 = DiscreteFactor(["x3", "x4", "x1"], [2, 2, 2], range(8))
        self.phi3 = DiscreteFactor(["x5", "x6", "x7"], [2, 2, 2], range(8))
        self.phi4 = DiscreteFactor(["x5", "x7", "x8"], [2, 2, 2], range(8))

    def test_class_init(self):
        phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 3, 2], range(12))
        phi2 = DiscreteFactor(["x3", "x4", "x1"], [2, 2, 2], range(8))
        factor_set1 = FactorSet(phi1, phi2)
        self.assertEqual({phi1, phi2}, factor_set1.get_factors())

    def test_factorset_add_remove_factors(self):
        self.factor_set1 = FactorSet()
        self.factor_set1.add_factors(self.phi1, self.phi2)
        self.assertEqual({self.phi1, self.phi2}, self.factor_set1.get_factors())
        self.factor_set1.remove_factors(self.phi2)
        self.assertEqual({self.phi1}, self.factor_set1.get_factors())

    def test_factorset_product(self):
        factor_set1 = FactorSet(self.phi1, self.phi2)
        factor_set2 = FactorSet(self.phi3, self.phi4)
        factor_set3 = factor_set2.product(factor_set1, inplace=False)
        self.assertEqual(
            {self.phi1, self.phi2, self.phi3, self.phi4}, factor_set3.factors
        )

    def test_factorset_divide(self):
        phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 3, 2], range(1, 13))
        phi2 = DiscreteFactor(["x3", "x4", "x1"], [2, 2, 2], range(1, 9))
        factor_set1 = FactorSet(phi1, phi2)
        phi3 = DiscreteFactor(["x5", "x6", "x7"], [2, 2, 2], range(1, 9))
        phi4 = DiscreteFactor(["x5", "x7", "x8"], [2, 2, 2], range(1, 9))
        factor_set2 = FactorSet(phi3, phi4)
        factor_set3 = factor_set2.divide(factor_set1, inplace=False)
        self.assertEqual(
            {phi3, phi4, phi1.identity_factor() / phi1, phi2.identity_factor() / phi2},
            factor_set3.factors,
        )

    def test_factorset_marginalize_inplace(self):
        factor_set = FactorSet(self.phi1, self.phi2, self.phi3, self.phi4)
        factor_set.marginalize(["x1", "x5"], inplace=True)
        phi1_equivalent_in_factor_set = list(
            filter(lambda x: set(x.scope()) == {"x2", "x3"}, factor_set.factors)
        )[0]
        self.assertEqual(
            self.phi1.marginalize(["x1"], inplace=False), phi1_equivalent_in_factor_set
        )
        phi2_equivalent_in_factor_set = list(
            filter(lambda x: set(x.scope()) == {"x4", "x3"}, factor_set.factors)
        )[0]
        self.assertEqual(
            self.phi2.marginalize(["x1"], inplace=False), phi2_equivalent_in_factor_set
        )
        phi3_equivalent_in_factor_set = list(
            filter(lambda x: set(x.scope()) == {"x6", "x7"}, factor_set.factors)
        )[0]
        self.assertEqual(
            self.phi3.marginalize(["x5"], inplace=False), phi3_equivalent_in_factor_set
        )
        phi4_equivalent_in_factor_set = list(
            filter(lambda x: set(x.scope()) == {"x8", "x7"}, factor_set.factors)
        )[0]
        self.assertEqual(
            self.phi4.marginalize(["x5"], inplace=False), phi4_equivalent_in_factor_set
        )

    def test_factorset_marginalize_not_inplace(self):
        factor_set = FactorSet(self.phi1, self.phi2, self.phi3, self.phi4)
        new_factor_set = factor_set.marginalize(["x1", "x5"], inplace=False)
        phi1_equivalent_in_factor_set = list(
            filter(lambda x: set(x.scope()) == {"x2", "x3"}, new_factor_set.factors)
        )[0]
        self.assertEqual(
            self.phi1.marginalize(["x1"], inplace=False), phi1_equivalent_in_factor_set
        )
        phi2_equivalent_in_factor_set = list(
            filter(lambda x: set(x.scope()) == {"x4", "x3"}, new_factor_set.factors)
        )[0]
        self.assertEqual(
            self.phi2.marginalize(["x1"], inplace=False), phi2_equivalent_in_factor_set
        )
        phi3_equivalent_in_factor_set = list(
            filter(lambda x: set(x.scope()) == {"x6", "x7"}, new_factor_set.factors)
        )[0]
        self.assertEqual(
            self.phi3.marginalize(["x5"], inplace=False), phi3_equivalent_in_factor_set
        )
        phi4_equivalent_in_factor_set = list(
            filter(lambda x: set(x.scope()) == {"x8", "x7"}, new_factor_set.factors)
        )[0]
        self.assertEqual(
            self.phi4.marginalize(["x5"], inplace=False), phi4_equivalent_in_factor_set
        )
