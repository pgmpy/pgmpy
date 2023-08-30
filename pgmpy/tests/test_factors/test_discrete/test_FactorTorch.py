import unittest
import warnings
from collections import OrderedDict
from shutil import get_terminal_size

import numpy as np
import numpy.testing as np_test

from pgmpy import config
from pgmpy.factors import factor_divide, factor_product, factor_sum_product
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete import JointProbabilityDistribution as JPD
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.independencies import Independencies
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork, MarkovNetwork
from pgmpy.utils import compat_fns, get_example_model


class TestFactorInitTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

    def test_class_init(self):
        phi = DiscreteFactor(
            variables=["x1", "x2", "x3"], cardinality=[2, 2, 2], values=np.ones(8)
        )
        self.assertEqual(phi.variables, ["x1", "x2", "x3"])
        np_test.assert_array_equal(phi.cardinality, np.array([2, 2, 2]))
        np_test.assert_array_equal(
            compat_fns.to_numpy(phi.values), np.ones(8).reshape(2, 2, 2)
        )

    def test_class_init_int_var(self):
        phi = DiscreteFactor(
            variables=[1, 2, 3], cardinality=[2, 3, 2], values=np.arange(12)
        )
        self.assertEqual(phi.variables, [1, 2, 3])
        np_test.assert_array_equal(phi.cardinality, np.array([2, 3, 2]))
        np_test.assert_array_equal(
            compat_fns.to_numpy(phi.values), np.arange(12).reshape(2, 3, 2)
        )

    def test_class_init_statenames(self):
        phi = DiscreteFactor(
            variables=["x1", "x2", "x3"],
            cardinality=[2, 2, 2],
            values=np.ones(8),
            state_names={
                "x1": ["sn0", "sn1"],
                "x2": ["sn0", "sn1"],
                "x3": ["sn0", "sn1"],
            },
        )
        self.assertEqual(phi.variables, ["x1", "x2", "x3"])
        np_test.assert_array_equal(phi.cardinality, np.array([2, 2, 2]))
        np_test.assert_array_equal(
            compat_fns.to_numpy(phi.values), np.ones(8).reshape(2, 2, 2)
        )

    def test_class_init_repeated_statename(self):
        self.assertRaises(
            ValueError,
            DiscreteFactor,
            ["x1", "x2", "x3"],
            [2, 2, 2],
            np.ones(8),
            {"x1": ["sn0", "sn0"], "x2": ["sn0", "sn1"], "x3": ["sn0", "sn1"]},
        )

    def test_class_init_nonlist_statename(self):
        self.assertRaises(
            ValueError,
            DiscreteFactor,
            ["x1", "x2", "x3"],
            [2, 2, 2],
            np.ones(8),
            {"x1": {"sn0", "sn0"}, "x2": ["sn0", "sn1"], "x3": ["sn0", "sn1"]},
        )

    def test_class_init_sizeerror(self):
        self.assertRaises(
            ValueError, DiscreteFactor, ["x1", "x2", "x3"], [2, 2, 2], np.ones(9)
        )

    def test_class_init_typeerror(self):
        self.assertRaises(TypeError, DiscreteFactor, "x1", [3], [1, 2, 3])
        self.assertRaises(
            ValueError, DiscreteFactor, ["x1", "x1", "x3"], [2, 3, 2], range(12)
        )

    def test_init_size_var_card_not_equal(self):
        self.assertRaises(ValueError, DiscreteFactor, ["x1", "x2"], [2], np.ones(2))

    def tearDown(self):
        config.set_backend("numpy")


class TestFactorMethodsTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        self.phi = DiscreteFactor(
            variables=["x1", "x2", "x3"],
            cardinality=[2, 2, 2],
            values=np.random.uniform(5, 10, size=8),
        )
        self.phi_sn = DiscreteFactor(
            variables=["x1", "x2", "x3"],
            cardinality=[2, 2, 2],
            values=np.random.uniform(5, 10, size=8),
            state_names={
                "x1": ("sn0", "sn1"),
                "x2": ("sn0", "sn1"),
                "x3": ("sn0", "sn1"),
            },
        )

        self.phi1 = DiscreteFactor(
            variables=["x1", "x2", "x3"], cardinality=[2, 3, 2], values=range(12)
        )
        self.phi1_sn = DiscreteFactor(
            variables=["x1", "x2", "x3"],
            cardinality=[2, 3, 2],
            values=range(12),
            state_names={
                "x1": ("sn0", "sn1"),
                "x2": ("sn0", "sn1", "sn2"),
                "x3": ("sn0", "sn1"),
            },
        )

        self.phi2 = DiscreteFactor(
            variables=[("x1", 0), ("x2", 0), ("x3", 0)],
            cardinality=[2, 3, 2],
            values=range(12),
        )

        self.phi2_sn = DiscreteFactor(
            variables=[("x1", 0), ("x2", 0), ("x3", 0)],
            cardinality=[2, 3, 2],
            values=range(12),
            state_names={
                "x1": ("sn0", "sn1"),
                "x2": ("sn0", "sn1", "sn2"),
                "x3": ("sn0", "sn1"),
            },
        )

        # This larger factor (phi3) caused a bug in reduce
        card3 = [3, 3, 3, 2, 2, 2, 2, 2, 2]
        self.phi3 = DiscreteFactor(
            ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            card3,
            np.arange(np.prod(card3), dtype=float),
        )

        self.tup1 = ("x1", "x2")
        self.tup2 = ("x2", "x3")
        self.tup3 = ("x3", (1, "x4"))
        self.phi4 = DiscreteFactor(
            [self.tup1, self.tup2, self.tup3],
            [2, 3, 4],
            np.random.uniform(3, 10, size=24),
        )
        self.phi5 = DiscreteFactor(
            [self.tup1, self.tup2, self.tup3], [2, 3, 4], range(24)
        )

        self.card6 = [4, 2, 1, 3, 5, 6]
        self.phi6 = DiscreteFactor(
            [
                self.tup1,
                self.tup2,
                self.tup3,
                self.tup1 + self.tup2,
                self.tup2 + self.tup3,
                self.tup3 + self.tup1,
            ],
            self.card6,
            np.arange(np.prod(self.card6), dtype=float),
        )

        self.var1 = "x1"
        self.var2 = ("x2", 1)
        self.var3 = frozenset(["x1", "x2"])
        self.phi7 = DiscreteFactor([self.var1, self.var2], [3, 2], [3, 2, 4, 5, 9, 8])
        self.phi8 = DiscreteFactor([self.var2, self.var3], [2, 2], [2, 1, 5, 6])
        self.phi9 = DiscreteFactor([self.var1, self.var3], [3, 2], [3, 2, 4, 5, 9, 8])
        self.phi10 = DiscreteFactor([self.var3], [2], [3, 6])

    def test_scope(self):
        self.assertListEqual(self.phi.scope(), ["x1", "x2", "x3"])
        self.assertListEqual(self.phi_sn.scope(), ["x1", "x2", "x3"])

        self.assertListEqual(self.phi1.scope(), ["x1", "x2", "x3"])
        self.assertListEqual(self.phi1_sn.scope(), ["x1", "x2", "x3"])

        self.assertListEqual(self.phi4.scope(), [self.tup1, self.tup2, self.tup3])

    def test_assignment(self):
        self.assertListEqual(
            self.phi.assignment([0]), [[("x1", 0), ("x2", 0), ("x3", 0)]]
        )
        self.assertListEqual(
            self.phi_sn.assignment([0]), [[("x1", "sn0"), ("x2", "sn0"), ("x3", "sn0")]]
        )

        self.assertListEqual(
            self.phi.assignment([4, 5, 6]),
            [
                [("x1", 1), ("x2", 0), ("x3", 0)],
                [("x1", 1), ("x2", 0), ("x3", 1)],
                [("x1", 1), ("x2", 1), ("x3", 0)],
            ],
        )

        self.assertListEqual(
            self.phi_sn.assignment([4, 5, 6]),
            [
                [("x1", "sn1"), ("x2", "sn0"), ("x3", "sn0")],
                [("x1", "sn1"), ("x2", "sn0"), ("x3", "sn1")],
                [("x1", "sn1"), ("x2", "sn1"), ("x3", "sn0")],
            ],
        )

        self.assertListEqual(
            self.phi1.assignment(np.array([4, 5, 6])),
            [
                [("x1", 0), ("x2", 2), ("x3", 0)],
                [("x1", 0), ("x2", 2), ("x3", 1)],
                [("x1", 1), ("x2", 0), ("x3", 0)],
            ],
        )

        self.assertListEqual(
            self.phi1_sn.assignment(np.array([4, 5, 6])),
            [
                [("x1", "sn0"), ("x2", "sn2"), ("x3", "sn0")],
                [("x1", "sn0"), ("x2", "sn2"), ("x3", "sn1")],
                [("x1", "sn1"), ("x2", "sn0"), ("x3", "sn0")],
            ],
        )

        self.assertListEqual(
            self.phi4.assignment(np.array([11, 12, 23])),
            [
                [(self.tup1, 0), (self.tup2, 2), (self.tup3, 3)],
                [(self.tup1, 1), (self.tup2, 0), (self.tup3, 0)],
                [(self.tup1, 1), (self.tup2, 2), (self.tup3, 3)],
            ],
        )

    def test_assignment_indexerror(self):
        self.assertRaises(IndexError, self.phi.assignment, [10])
        self.assertRaises(IndexError, self.phi_sn.assignment, [10])
        self.assertRaises(IndexError, self.phi.assignment, [1, 3, 10, 5])
        self.assertRaises(IndexError, self.phi_sn.assignment, [1, 3, 10, 5])
        self.assertRaises(IndexError, self.phi.assignment, np.array([1, 3, 10, 5]))
        self.assertRaises(IndexError, self.phi_sn.assignment, np.array([1, 3, 10, 5]))

        self.assertRaises(IndexError, self.phi4.assignment, [2, 24])
        self.assertRaises(IndexError, self.phi4.assignment, np.array([24, 2, 4, 30]))

    def test_get_cardinality(self):
        self.assertEqual(self.phi.get_cardinality(["x1"]), {"x1": 2})
        self.assertEqual(self.phi_sn.get_cardinality(["x1"]), {"x1": 2})

        self.assertEqual(self.phi.get_cardinality(["x2"]), {"x2": 2})
        self.assertEqual(self.phi_sn.get_cardinality(["x2"]), {"x2": 2})

        self.assertEqual(self.phi.get_cardinality(["x3"]), {"x3": 2})
        self.assertEqual(self.phi_sn.get_cardinality(["x3"]), {"x3": 2})

        self.assertEqual(self.phi.get_cardinality(["x1", "x2"]), {"x1": 2, "x2": 2})
        self.assertEqual(self.phi_sn.get_cardinality(["x1", "x2"]), {"x1": 2, "x2": 2})

        self.assertEqual(self.phi.get_cardinality(["x1", "x3"]), {"x1": 2, "x3": 2})
        self.assertEqual(self.phi_sn.get_cardinality(["x1", "x3"]), {"x1": 2, "x3": 2})

        self.assertEqual(
            self.phi.get_cardinality(["x1", "x2", "x3"]), {"x1": 2, "x2": 2, "x3": 2}
        )
        self.assertEqual(
            self.phi_sn.get_cardinality(["x1", "x2", "x3"]), {"x1": 2, "x2": 2, "x3": 2}
        )

        self.assertEqual(
            self.phi4.get_cardinality([self.tup1, self.tup3]),
            {self.tup1: 2, self.tup3: 4},
        )

    def test_get_cardinality_scopeerror(self):
        self.assertRaises(ValueError, self.phi.get_cardinality, ["x4"])
        self.assertRaises(ValueError, self.phi4.get_cardinality, [("x1", "x4")])

        self.assertRaises(ValueError, self.phi4.get_cardinality, [("x3", (2, "x4"))])

    def test_get_cardinality_typeerror(self):
        self.assertRaises(TypeError, self.phi.get_cardinality, "x1")
        self.assertRaises(TypeError, self.phi_sn.get_cardinality, "x1")

    def test_get_value(self):
        model = get_example_model("asia")
        cpd = model.get_cpds("either")

        for phi in [cpd, cpd.to_factor()]:
            phi_copy = phi.copy()
            self.assertEqual(phi.get_value(lung="yes", tub="no", either="yes"), 1.0)
            self.assertEqual(phi.get_value(lung=0, tub=1, either=0), 1.0)
            self.assertEqual(phi.get_value(lung="yes", tub=1, either="yes"), 1.0)
            self.assertRaises(ValueError, phi.get_value, lung="yes", either="yes")
            self.assertRaises(
                ValueError, phi.get_value, lung="yes", tub="no", boo="yes"
            )
            self.assertEqual(phi, phi_copy)

    def test_set_value(self):
        model = get_example_model("asia")
        cpd = model.get_cpds("either")
        for phi in [cpd, cpd.to_factor()]:
            phi_copy = phi.copy()

            phi.set_value(value=0.1, lung="yes", tub="no", either="yes")
            self.assertEqual(phi.get_value(lung="yes", tub="no", either="yes"), 0.1)

            phi.set_value(value=0.2, lung=0, tub=1, either=0)
            self.assertEqual(phi.get_value(lung=0, tub=1, either=0), 0.2)

            phi.set_value(value=0.3, lung="yes", tub=1, either="yes")
            self.assertEqual(phi.get_value(lung="yes", tub=1, either="yes"), 0.3)

            phi.set_value(value=5, lung="yes", tub=1, either="yes")
            self.assertEqual(phi.get_value(lung="yes", tub=1, either="yes"), 5)

            self.assertRaises(
                ValueError, phi.set_value, value=0.1, lung="yes", either="yes"
            )
            self.assertRaises(
                ValueError, phi.set_value, value=0.1, lung="yes", tub="no", boo="yes"
            )
            self.assertRaises(
                ValueError, phi.set_value, value="a", lung="yes", tub="no", either="yes"
            )
            self.assertRaises(
                ValueError,
                phi.set_value,
                value=None,
                lung="yes",
                tub="no",
                either="yes",
            )
            self.assertNotEqual(phi, phi_copy)

    def test_marginalize(self):
        self.phi1.marginalize(["x1"])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi1.values),
            np.array([[6, 8], [10, 12], [14, 16]]),
        )
        self.phi1_sn.marginalize(["x1"])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi1_sn.values),
            np.array([[6, 8], [10, 12], [14, 16]]),
        )

        self.phi1.marginalize(["x2"])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi1.values), np.array([30, 36])
        )
        self.phi1_sn.marginalize(["x2"])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi1_sn.values), np.array([30, 36])
        )

        self.phi1.marginalize(["x3"])
        np_test.assert_array_equal(compat_fns.to_numpy(self.phi1.values), np.array(66))
        self.phi1_sn.marginalize(["x3"])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi1_sn.values), np.array(66)
        )

        self.phi5.marginalize([self.tup1])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi5.values),
            np.array([[12, 14, 16, 18], [20, 22, 24, 26], [28, 30, 32, 34]]),
        )
        self.phi5.marginalize([self.tup2])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi5.values), np.array([60, 66, 72, 78])
        )

        self.phi5.marginalize([self.tup3])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi5.values), np.array([276])
        )

    def test_marginalize_scopeerror(self):
        self.assertRaises(ValueError, self.phi.marginalize, ["x4"])
        self.phi.marginalize(["x1"])
        self.assertRaises(ValueError, self.phi.marginalize, ["x1"])

        self.assertRaises(ValueError, self.phi4.marginalize, [("x1", "x3")])
        self.phi4.marginalize([self.tup2])
        self.assertRaises(ValueError, self.phi4.marginalize, [self.tup2])

    def test_marginalize_typeerror(self):
        self.assertRaises(TypeError, self.phi.marginalize, "x1")

    def test_marginalize_shape(self):
        values = ["A", "D", "F", "H"]
        phi3_mar = self.phi3.marginalize(values, inplace=False)
        # Previously a sorting error caused these to be different
        np_test.assert_array_equal(
            compat_fns.to_numpy(phi3_mar.values.shape), phi3_mar.cardinality
        )

        phi6_mar = self.phi6.marginalize([self.tup1, self.tup2], inplace=False)
        np_test.assert_array_equal(
            compat_fns.to_numpy(phi6_mar.values.shape), phi6_mar.cardinality
        )

        self.phi6.marginalize([self.tup1, self.tup3 + self.tup1], inplace=True)
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi6.values.shape), self.phi6.cardinality
        )

    def test_normalize(self):
        self.phi1.normalize()
        np_test.assert_almost_equal(
            compat_fns.to_numpy(self.phi1.values),
            np.array(
                [
                    [
                        [0, 0.01515152],
                        [0.03030303, 0.04545455],
                        [0.06060606, 0.07575758],
                    ],
                    [
                        [0.09090909, 0.10606061],
                        [0.12121212, 0.13636364],
                        [0.15151515, 0.16666667],
                    ],
                ]
            ),
        )

        self.phi1_sn.normalize()
        np_test.assert_almost_equal(
            compat_fns.to_numpy(self.phi1_sn.values),
            np.array(
                [
                    [
                        [0, 0.01515152],
                        [0.03030303, 0.04545455],
                        [0.06060606, 0.07575758],
                    ],
                    [
                        [0.09090909, 0.10606061],
                        [0.12121212, 0.13636364],
                        [0.15151515, 0.16666667],
                    ],
                ]
            ),
        )

        self.phi5.normalize()
        np_test.assert_almost_equal(
            compat_fns.to_numpy(self.phi5.values),
            [
                [
                    [0.0, 0.00362319, 0.00724638, 0.01086957],
                    [0.01449275, 0.01811594, 0.02173913, 0.02536232],
                    [0.02898551, 0.0326087, 0.03623188, 0.03985507],
                ],
                [
                    [0.04347826, 0.04710145, 0.05072464, 0.05434783],
                    [0.05797101, 0.0615942, 0.06521739, 0.06884058],
                    [0.07246377, 0.07608696, 0.07971014, 0.08333333],
                ],
            ],
        )

    def test_reduce(self):
        self.phi1.reduce([("x1", 0), ("x2", 0)])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi1.values), np.array([0, 1])
        )
        self.phi1_sn.reduce([("x1", "sn0"), ("x2", "sn0")])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi1_sn.values), np.array([0, 1])
        )

        self.phi5.reduce([(self.tup1, 0), (self.tup3, 1)])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi5.values), np.array([1, 5, 9])
        )

    def test_reduce1(self):
        self.phi1.reduce([("x2", 0), ("x1", 0)])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi1.values), np.array([0, 1])
        )
        self.phi1_sn.reduce([("x2", "sn0"), ("x1", "sn0")])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi1_sn.values), np.array([0, 1])
        )

        self.phi5.reduce([(self.tup3, 1), (self.tup1, 0)])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi5.values), np.array([1, 5, 9])
        )

    def test_reduce_shape(self):
        values = [("A", 0), ("D", 0), ("F", 0), ("H", 1)]
        phi3_reduced = self.phi3.reduce(values, inplace=False)
        # Previously a sorting error caused these to be different
        np_test.assert_array_equal(
            compat_fns.to_numpy(phi3_reduced.values.shape), phi3_reduced.cardinality
        )

        values = [(self.tup1, 2), (self.tup3, 0)]
        phi6_reduced = self.phi6.reduce(values, inplace=False)
        np_test.assert_array_equal(
            compat_fns.to_numpy(phi6_reduced.values.shape), phi6_reduced.cardinality
        )

        self.phi6.reduce(values, inplace=True)
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi6.values.shape), self.phi6.cardinality
        )

    def test_complete_reduce(self):
        self.phi1.reduce([("x1", 0), ("x2", 0), ("x3", 1)])
        np_test.assert_array_equal(compat_fns.to_numpy(self.phi1.values), np.array([1]))
        np_test.assert_array_equal(self.phi1.cardinality, np.array([]))
        np_test.assert_array_equal(self.phi1.variables, OrderedDict())

        self.phi1_sn.reduce([("x1", "sn0"), ("x2", "sn0"), ("x3", "sn1")])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi1_sn.values), np.array([1])
        )
        np_test.assert_array_equal(self.phi1_sn.cardinality, np.array([]))
        np_test.assert_array_equal(self.phi1_sn.variables, OrderedDict())

        self.phi5.reduce([(("x1", "x2"), 1), (("x2", "x3"), 0), (("x3", (1, "x4")), 3)])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.phi5.values), np.array([15])
        )
        np_test.assert_array_equal(self.phi5.cardinality, np.array([]))
        np_test.assert_array_equal(self.phi5.variables, OrderedDict())

    def test_reduce_typeerror(self):
        self.assertRaises(TypeError, self.phi1.reduce, "x10")
        self.assertRaises(TypeError, self.phi1.reduce, ["x10"])

    def test_reduce_scopeerror(self):
        self.assertRaises(ValueError, self.phi1.reduce, [("x4", 1)])
        self.assertRaises(ValueError, self.phi5.reduce, [((("x1", 0.1), 0))])

    def test_reduce_sizeerror(self):
        self.assertRaises(IndexError, self.phi1.reduce, [("x3", 5)])
        self.assertRaises(IndexError, self.phi5.reduce, [(("x2", "x3"), 3)])

    def test_identity_factor(self):
        identity_factor = self.phi.identity_factor()
        self.assertEqual(list(identity_factor.variables), ["x1", "x2", "x3"])
        np_test.assert_array_equal(identity_factor.cardinality, [2, 2, 2])
        np_test.assert_array_equal(
            compat_fns.to_numpy(identity_factor.values), np.ones(8).reshape(2, 2, 2)
        )

        identity_factor1 = self.phi5.identity_factor()
        self.assertEqual(
            list(identity_factor1.variables), [self.tup1, self.tup2, self.tup3]
        )
        np_test.assert_array_equal(identity_factor1.cardinality, [2, 3, 4])
        np_test.assert_array_equal(
            compat_fns.to_numpy(identity_factor1.values), np.ones(24).reshape(2, 3, 4)
        )

    def test_factor_product(self):
        phi = DiscreteFactor(["x1", "x2"], [2, 2], range(4))
        phi1 = DiscreteFactor(["x3", "x4"], [2, 2], range(4))
        prod = factor_product(phi, phi1)
        expected_factor = DiscreteFactor(
            ["x1", "x2", "x3", "x4"],
            [2, 2, 2, 2],
            [0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9],
        )
        self.assertEqual(prod, expected_factor)
        self.assertEqual(sorted(prod.variables), ["x1", "x2", "x3", "x4"])

        phi = DiscreteFactor(["x1", "x2"], [3, 2], range(6))
        phi1 = DiscreteFactor(["x2", "x3"], [2, 2], range(4))
        prod = factor_product(phi, phi1)
        expected_factor = DiscreteFactor(
            ["x1", "x2", "x3"], [3, 2, 2], [0, 0, 2, 3, 0, 2, 6, 9, 0, 4, 10, 15]
        )
        self.assertEqual(prod, expected_factor)
        self.assertEqual(set(prod.variables), set(expected_factor.variables))

        prod = factor_product(self.phi7, self.phi8)
        expected_factor = DiscreteFactor(
            [self.var1, self.var2, self.var3],
            [3, 2, 2],
            [6, 3, 10, 12, 8, 4, 25, 30, 18, 9, 40, 48],
        )
        self.assertEqual(prod, expected_factor)
        self.assertEqual(set(prod.variables), set(expected_factor.variables))

        # Test for Issue 1565
        prod = factor_product(phi)
        self.assertEqual(phi, prod)
        self.assertNotEqual(id(phi), id(prod))

    def test_product(self):
        phi = DiscreteFactor(["x1", "x2"], [2, 2], range(4))
        phi1 = DiscreteFactor(["x3", "x4"], [2, 2], range(4))
        prod = phi.product(phi1, inplace=False)
        expected_factor = DiscreteFactor(
            ["x1", "x2", "x3", "x4"],
            [2, 2, 2, 2],
            [0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9],
        )
        self.assertEqual(prod, expected_factor)
        self.assertEqual(sorted(prod.variables), ["x1", "x2", "x3", "x4"])

        phi = DiscreteFactor(["x1", "x2"], [3, 2], range(6))
        phi1 = DiscreteFactor(["x2", "x3"], [2, 2], range(4))
        prod = phi.product(phi1, inplace=False)
        expected_factor = DiscreteFactor(
            ["x1", "x2", "x3"], [3, 2, 2], [0, 0, 2, 3, 0, 2, 6, 9, 0, 4, 10, 15]
        )
        self.assertEqual(prod, expected_factor)
        self.assertEqual(sorted(prod.variables), ["x1", "x2", "x3"])

        phi7_copy = self.phi7
        phi7_copy.product(self.phi8, inplace=True)
        expected_factor = DiscreteFactor(
            [self.var1, self.var2, self.var3],
            [3, 2, 2],
            [6, 3, 10, 12, 8, 4, 25, 30, 18, 9, 40, 48],
        )
        self.assertEqual(expected_factor, phi7_copy)
        self.assertEqual(
            set(phi7_copy.variables), set([self.var1, self.var2, self.var3])
        )

    def test_factor_product_non_factor_arg(self):
        self.assertRaises(TypeError, factor_product, 1, 2)

    def test_factor_mul(self):
        phi = DiscreteFactor(["x1", "x2"], [2, 2], range(4))
        phi1 = DiscreteFactor(["x3", "x4"], [2, 2], range(4))
        prod = phi * phi1

        sorted_vars = ["x1", "x2", "x3", "x4"]
        for axis in range(prod.values.ndim):
            exchange_index = prod.variables.index(sorted_vars[axis])
            prod.variables[axis], prod.variables[exchange_index] = (
                prod.variables[exchange_index],
                prod.variables[axis],
            )
            prod.values = prod.values.swapaxes(axis, exchange_index)

        np_test.assert_almost_equal(
            compat_fns.to_numpy(prod.values.ravel()),
            np.array([0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9]),
        )

        self.assertEqual(prod.variables, ["x1", "x2", "x3", "x4"])

    def test_factor_divide(self):
        phi1 = DiscreteFactor(["x1", "x2"], [2, 2], [1, 2, 2, 4])
        phi2 = DiscreteFactor(["x1"], [2], [1, 2])
        expected_factor = phi1.divide(phi2, inplace=False)
        phi3 = DiscreteFactor(["x1", "x2"], [2, 2], [1, 2, 1, 2])
        self.assertEqual(phi3, expected_factor)

        self.phi9.divide(self.phi10, inplace=True)
        np_test.assert_array_almost_equal(
            compat_fns.to_numpy(self.phi9.values),
            np.array(
                [1.000000, 0.333333, 1.333333, 0.833333, 3.000000, 1.333333]
            ).reshape(3, 2),
        )
        self.assertEqual(self.phi9.variables, [self.var1, self.var3])

    def test_factor_divide_truediv(self):
        phi1 = DiscreteFactor(["x1", "x2"], [2, 2], [1, 2, 2, 4])
        phi2 = DiscreteFactor(["x1"], [2], [1, 2])
        div = phi1 / phi2
        phi3 = DiscreteFactor(["x1", "x2"], [2, 2], [1, 2, 1, 2])
        self.assertEqual(phi3, div)

        self.phi9 = self.phi9 / self.phi10
        np_test.assert_array_almost_equal(
            compat_fns.to_numpy(self.phi9.values),
            np.array(
                [1.000000, 0.333333, 1.333333, 0.833333, 3.000000, 1.333333]
            ).reshape(3, 2),
        )
        self.assertEqual(self.phi9.variables, [self.var1, self.var3])

    def test_factor_divide_invalid(self):
        phi1 = DiscreteFactor(["x1", "x2"], [2, 2], [1, 2, 3, 4])
        phi2 = DiscreteFactor(["x1"], [2], [0, 2])
        div = phi1.divide(phi2, inplace=False)
        np_test.assert_array_equal(
            compat_fns.to_numpy(div.values.ravel()), np.array([np.inf, np.inf, 1.5, 2])
        )

    def test_factor_divide_no_common_scope(self):
        phi1 = DiscreteFactor(["x1", "x2"], [2, 2], [1, 2, 3, 4])
        phi2 = DiscreteFactor(["x3"], [2], [0, 2])
        self.assertRaises(ValueError, factor_divide, phi1, phi2)

        phi2 = DiscreteFactor([self.var3], [2], [2, 1])
        self.assertRaises(ValueError, factor_divide, self.phi7, phi2)

    def test_factor_divide_non_factor_arg(self):
        self.assertRaises(TypeError, factor_divide, 1, 1)

    def test_factor_sum_product(self):
        model = get_example_model("alarm")
        infer = VariableElimination(model)
        phi = [cpd.to_factor() for cpd in model.cpds]
        phi_history = factor_sum_product(output_vars=["HISTORY"], factors=phi)
        self.assertEqual(infer.query(["HISTORY"]), phi_history)

    def test_factor_sum(self):
        phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 3, 2], range(12))
        phi2 = DiscreteFactor(["x3", "x4", "x1"], [2, 2, 2], range(8))
        phi1.sum(phi2, inplace=True)

        self.assertEqual(sorted(phi1.variables), ["x1", "x2", "x3", "x4"])
        self.assertEqual(
            phi1.get_cardinality(phi1.variables), {"x1": 2, "x2": 3, "x3": 2, "x4": 2}
        )
        np_test.assert_almost_equal(
            compat_fns.to_numpy(phi1.values),
            np.array(
                [
                    [
                        [[0.0, 2.0], [5.0, 7.0]],
                        [[2.0, 4.0], [7.0, 9.0]],
                        [[4.0, 6.0], [9.0, 11.0]],
                    ],
                    [
                        [[7.0, 9.0], [12.0, 14.0]],
                        [[9.0, 11.0], [14.0, 16.0]],
                        [[11.0, 13.0], [16.0, 18.0]],
                    ],
                ]
            ),
        )

        phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 3, 2], range(12))
        phi1.sum(2, inplace=True)
        self.assertEqual(phi1.variables, ["x1", "x2", "x3"])
        self.assertEqual(
            phi1.get_cardinality(phi1.variables), {"x1": 2, "x2": 3, "x3": 2}
        )
        np_test.assert_almost_equal(
            compat_fns.to_numpy(phi1.values), np.arange(2, 14).reshape(2, 3, 2)
        )

    def test_eq(self):
        self.assertFalse(self.phi == self.phi1)
        self.assertTrue(self.phi == self.phi)
        self.assertTrue(self.phi1 == self.phi1)

        self.assertTrue(self.phi5 == self.phi5)
        self.assertFalse(self.phi5 == self.phi6)
        self.assertTrue(self.phi6 == self.phi6)

    def test_eq_state_names_order(self):
        phi = DiscreteFactor(
            ["x", "y"],
            [2, 2],
            [0.1, 0.1, 0.9, 0.9],
            state_names={"x": ["x1", "x2"], "y": ["y1", "y2"]},
        )
        phi2 = DiscreteFactor(
            ["x", "y"],
            [2, 2],
            [0.9, 0.9, 0.1, 0.1],
            state_names={"x": ["x2", "x1"], "y": ["y2", "y1"]},
        )

        self.assertTrue(phi == phi2)

    def test_eq1(self):
        phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 4, 3], range(24))
        phi2 = DiscreteFactor(
            ["x2", "x1", "x3"],
            [4, 2, 3],
            [
                0,
                1,
                2,
                12,
                13,
                14,
                3,
                4,
                5,
                15,
                16,
                17,
                6,
                7,
                8,
                18,
                19,
                20,
                9,
                10,
                11,
                21,
                22,
                23,
            ],
        )
        self.assertTrue(phi1 == phi2)
        self.assertEqual(phi2.variables, ["x2", "x1", "x3"])

        phi3 = DiscreteFactor([self.tup1, self.tup2, self.tup3], [2, 4, 3], range(24))
        phi4 = DiscreteFactor(
            [self.tup2, self.tup1, self.tup3],
            [4, 2, 3],
            [
                0,
                1,
                2,
                12,
                13,
                14,
                3,
                4,
                5,
                15,
                16,
                17,
                6,
                7,
                8,
                18,
                19,
                20,
                9,
                10,
                11,
                21,
                22,
                23,
            ],
        )
        self.assertTrue(phi3 == phi4)

    def test_sample(self):
        phi1 = DiscreteFactor(["x1", "x2"], [2, 2], [1, 2, 3, 4])
        samples = phi1.sample(int(1e5))
        self.assertEqual(samples.shape, (1e5, 2))
        np_test.assert_almost_equal(
            (samples.groupby(["x1", "x2"]).size() / int(1e5)).values,
            np.array([1, 2, 3, 4]) / 10,
            decimal=2,
        )

        phi1 = DiscreteFactor(
            ["x1", "x2"],
            [2, 2],
            [1, 2, 3, 4],
            state_names={"x1": ["a1", "a2"], "x2": ["b1", "b2"]},
        )
        samples = phi1.sample(int(1e5))
        self.assertEqual(samples.shape, (1e5, 2))
        np_test.assert_almost_equal(
            (samples.groupby(["x1", "x2"]).size() / int(1e5)).values,
            np.array([1, 2, 3, 4]) / 10,
            decimal=2,
        )

    def test_hash(self):
        phi1 = DiscreteFactor(["x1", "x2"], [2, 2], [1, 2, 3, 4])
        phi2 = DiscreteFactor(["x2", "x1"], [2, 2], [1, 3, 2, 4])
        self.assertEqual(hash(phi1), hash(phi2))

        phi1 = DiscreteFactor(["x1", "x2", "x3"], [2, 2, 2], range(8))
        phi2 = DiscreteFactor(["x3", "x1", "x2"], [2, 2, 2], [0, 2, 4, 6, 1, 3, 5, 7])
        self.assertEqual(hash(phi1), hash(phi2))

        var1 = TestHash(1, 2)
        phi3 = DiscreteFactor([var1, self.var2, self.var3], [2, 4, 3], range(24))
        phi4 = DiscreteFactor(
            [self.var2, var1, self.var3],
            [4, 2, 3],
            [
                0,
                1,
                2,
                12,
                13,
                14,
                3,
                4,
                5,
                15,
                16,
                17,
                6,
                7,
                8,
                18,
                19,
                20,
                9,
                10,
                11,
                21,
                22,
                23,
            ],
        )
        self.assertEqual(hash(phi3), hash(phi4))

        var1 = TestHash(2, 3)
        var2 = TestHash("x2", 1)
        phi3 = DiscreteFactor([var1, var2, self.var3], [2, 2, 2], range(8))
        phi4 = DiscreteFactor(
            [self.var3, var1, var2], [2, 2, 2], [0, 2, 4, 6, 1, 3, 5, 7]
        )
        self.assertEqual(hash(phi3), hash(phi4))

    def test_maximize_single(self):
        self.phi1.maximize(["x1"])
        self.assertEqual(
            self.phi1, DiscreteFactor(["x2", "x3"], [3, 2], [6, 7, 8, 9, 10, 11])
        )
        self.phi1.maximize(["x2"])
        self.assertEqual(self.phi1, DiscreteFactor(["x3"], [2], [10, 11]))
        self.phi2 = DiscreteFactor(
            ["x1", "x2", "x3"],
            [3, 2, 2],
            [0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0.00, 0.00, 0.15, 0.21, 0.08, 0.18],
        )
        self.phi2.maximize(["x2"])
        self.assertEqual(
            self.phi2,
            DiscreteFactor(["x1", "x3"], [3, 2], [0.25, 0.35, 0.05, 0.07, 0.15, 0.21]),
        )

        self.phi5.maximize([("x1", "x2")])
        self.assertEqual(
            self.phi5,
            DiscreteFactor(
                [("x2", "x3"), ("x3", (1, "x4"))],
                [3, 4],
                [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            ),
        )
        self.phi5.maximize([("x2", "x3")])
        self.assertEqual(
            self.phi5, DiscreteFactor([("x3", (1, "x4"))], [4], [20, 21, 22, 23])
        )

    def test_maximize_list(self):
        self.phi1.maximize(["x1", "x2"])
        self.assertEqual(self.phi1, DiscreteFactor(["x3"], [2], [10, 11]))

        self.phi5.maximize([("x1", "x2"), ("x2", "x3")])
        self.assertEqual(
            self.phi5, DiscreteFactor([("x3", (1, "x4"))], [4], [20, 21, 22, 23])
        )

    def test_maximize_shape(self):
        values = ["A", "D", "F", "H"]
        phi3_max = self.phi3.maximize(values, inplace=False)
        # Previously a sorting error caused these to be different
        np_test.assert_array_equal(phi3_max.values.shape, phi3_max.cardinality)

        phi = DiscreteFactor(
            [self.var1, self.var2, self.var3],
            [3, 2, 2],
            [3, 2, 4, 5, 9, 8, 3, 2, 4, 5, 9, 8],
        )
        phi_max = phi.marginalize([self.var1, self.var2], inplace=False)
        np_test.assert_array_equal(phi_max.values.shape, phi_max.cardinality)

    def test_maximize_scopeerror(self):
        self.assertRaises(ValueError, self.phi.maximize, ["x10"])

    def test_maximize_typeerror(self):
        self.assertRaises(TypeError, self.phi.maximize, "x1")

    def tearDown(self):
        del self.phi
        del self.phi1
        del self.phi2
        del self.phi3
        del self.phi4
        del self.phi5
        del self.phi6
        del self.phi7
        del self.phi8
        del self.phi9
        del self.phi10

        config.set_backend("numpy")


class TestHash:
    # Used to check the hash function of DiscreteFactor class.

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash(str(self.x) + str(self.y))

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.x == other.x
            and self.y == other.y
        )


class TestTabularCPDInitTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

    def test_cpd_init(self):
        cpd = TabularCPD("grade", 3, [[0.1], [0.1], [0.1]])
        self.assertEqual(cpd.variable, "grade")
        self.assertEqual(cpd.variable_card, 3)
        self.assertEqual(list(cpd.variables), ["grade"])
        np_test.assert_array_equal(cpd.cardinality, np.array([3]))
        np_test.assert_array_almost_equal(
            compat_fns.to_numpy(cpd.values), np.array([0.1, 0.1, 0.1])
        )

        values = [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ]

        evidence = ["intel", "diff"]
        evidence_card = [3, 2]

        valid_value_inputs = [values, np.asarray(values)]
        valid_evidence_inputs = [evidence, set(evidence), np.asarray(evidence)]
        valid_evidence_card_inputs = [evidence_card, np.asarray(evidence_card)]

        for value in valid_value_inputs:
            for evidence in valid_evidence_inputs:
                for evidence_card in valid_evidence_card_inputs:
                    cpd = TabularCPD(
                        "grade",
                        3,
                        values,
                        evidence=["intel", "diff"],
                        evidence_card=[3, 2],
                    )
                    self.assertEqual(cpd.variable, "grade")
                    self.assertEqual(cpd.variable_card, 3)
                    np_test.assert_array_equal(cpd.cardinality, np.array([3, 3, 2]))
                    self.assertListEqual(
                        list(cpd.variables), ["grade", "intel", "diff"]
                    )
                    np_test.assert_almost_equal(
                        compat_fns.to_numpy(cpd.values),
                        np.array(
                            [
                                0.1,
                                0.1,
                                0.1,
                                0.1,
                                0.1,
                                0.1,
                                0.1,
                                0.1,
                                0.1,
                                0.1,
                                0.1,
                                0.1,
                                0.8,
                                0.8,
                                0.8,
                                0.8,
                                0.8,
                                0.8,
                            ]
                        ).reshape(3, 3, 2),
                    )

        cpd = TabularCPD(
            "grade",
            3,
            [[0.1, 0.1], [0.1, 0.1], [0.8, 0.8]],
            evidence=["evi1"],
            evidence_card=[2.0],
        )
        self.assertEqual(cpd.variable, "grade")
        self.assertEqual(cpd.variable_card, 3)
        np_test.assert_array_equal(cpd.cardinality, np.array([3, 2]))
        self.assertListEqual(list(cpd.variables), ["grade", "evi1"])
        np_test.assert_almost_equal(
            compat_fns.to_numpy(cpd.values),
            np.array([0.1, 0.1, 0.1, 0.1, 0.8, 0.8]).reshape(3, 2),
        )

    def test_cpd_init_event_card_not_int(self):
        self.assertRaises(TypeError, TabularCPD, "event", "2", [[0.1, 0.9]])

    def test_cpd_init_cardinality_not_specified(self):
        self.assertRaises(
            ValueError,
            TabularCPD,
            "event",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            ["evi1", "evi2"],
            [5],
        )
        self.assertRaises(
            ValueError,
            TabularCPD,
            "event",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            ["evi1", "evi2"],
            [5.0],
        )
        self.assertRaises(
            ValueError,
            TabularCPD,
            "event",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            ["evi1"],
            [5, 6],
        )
        self.assertRaises(
            TypeError,
            TabularCPD,
            "event",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            "evi1",
            [5, 6],
        )

    def test_cpd_init_value_not_2d(self):
        self.assertRaises(
            TypeError,
            TabularCPD,
            "event",
            3,
            [
                [
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                ]
            ],
            ["evi1", "evi2"],
            [5, 6],
        )

    def test_too_wide_cpd_table(self):
        terminal_width, terminal_height = get_terminal_size()

        grasp_cpd = TabularCPD(
            "grasp",
            4,
            [
                [
                    0,
                    0.016,
                    0.13,
                    0.3,
                    0.3,
                    0,
                    0.025,
                    0.2,
                    0.45,
                    0.5,
                    0,
                    0.025,
                    0.2,
                    0.45,
                    0.5,
                    0,
                    0.05,
                    0.4,
                    0.9,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.016,
                    0.1,
                    0.28,
                    0.3,
                    0,
                    0.025,
                    0.15,
                    0.425,
                    0.5,
                    0,
                    0.025,
                    0.15,
                    0.425,
                    0.5,
                    0,
                    0.05,
                    0.3,
                    0.85,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.006,
                    0.06,
                    0.273,
                    0.3,
                    0,
                    0.01,
                    0.1,
                    0.41,
                    0.5,
                    0,
                    0.01,
                    0.1,
                    0.41,
                    0.5,
                    0,
                    0.02,
                    0.2,
                    0.82,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.06,
                    0.26,
                    0.316,
                    0,
                    0,
                    0.1,
                    0.4,
                    0.475,
                    0,
                    0,
                    0.1,
                    0.4,
                    0.475,
                    0,
                    0,
                    0.2,
                    0.8,
                    0.95,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0.017,
                    0.13,
                    0.3,
                    0.3,
                    0,
                    0.025,
                    0.2,
                    0.45,
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.025,
                    0.2,
                    0.45,
                    0.5,
                    0,
                    0.05,
                    0.4,
                    0.9,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.017,
                    0.1,
                    0.28,
                    0.3,
                    0,
                    0.025,
                    0.15,
                    0.425,
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.025,
                    0.15,
                    0.425,
                    0.5,
                    0,
                    0.05,
                    0.3,
                    0.85,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.007,
                    0.07,
                    0.273,
                    0.3,
                    0,
                    0.01,
                    0.1,
                    0.41,
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.41,
                    0.5,
                    0,
                    0.02,
                    0.2,
                    0.82,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.07,
                    0.27,
                    0.317,
                    0,
                    0,
                    0.1,
                    0.4,
                    0.475,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.1,
                    0.4,
                    0.475,
                    0,
                    0,
                    0.2,
                    0.8,
                    0.95,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0.017,
                    0.14,
                    0.3,
                    0.4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.025,
                    0.2,
                    0.45,
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.025,
                    0.2,
                    0.45,
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.05,
                    0.4,
                    0.9,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.017,
                    0.1,
                    0.29,
                    0.4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.025,
                    0.15,
                    0.425,
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.025,
                    0.15,
                    0.425,
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.05,
                    0.3,
                    0.85,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.007,
                    0.07000000000000001,
                    0.274,
                    0.4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.41,
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.41,
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.02,
                    0.2,
                    0.82,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.07000000000000001,
                    0.27,
                    0.317,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.1,
                    0.4,
                    0.475,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.1,
                    0.4,
                    0.475,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.2,
                    0.8,
                    0.95,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0.95,
                    0.6,
                    0.1,
                    0,
                    1,
                    0.95,
                    0.6,
                    0.1,
                    0,
                    1,
                    0.95,
                    0.6,
                    0.1,
                    0,
                    1,
                    0.95,
                    0.6,
                    0.1,
                    0,
                    1,
                    0.95,
                    0.6,
                    0.1,
                    0,
                    1,
                    0.95,
                    0.6,
                    0.1,
                    0,
                    1,
                    0.95,
                    0.6,
                    0.1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0.95,
                    0.7,
                    0.15,
                    0,
                    1,
                    0.95,
                    0.7,
                    0.15,
                    0,
                    1,
                    0.95,
                    0.7,
                    0.15,
                    0,
                    1,
                    0.95,
                    0.7,
                    0.15,
                    0,
                    1,
                    0.95,
                    0.7,
                    0.15,
                    0,
                    1,
                    0.95,
                    0.7,
                    0.15,
                    0,
                    1,
                    0.95,
                    0.7,
                    0.15,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0.98,
                    0.8,
                    0.18,
                    0,
                    1,
                    0.98,
                    0.8,
                    0.18,
                    0,
                    1,
                    0.98,
                    0.8,
                    0.18,
                    0,
                    1,
                    0.98,
                    0.8,
                    0.18,
                    0,
                    1,
                    0.98,
                    0.8,
                    0.18,
                    0,
                    1,
                    0.98,
                    0.8,
                    0.18,
                    0,
                    1,
                    0.98,
                    0.8,
                    0.18,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0.8,
                    0.2,
                    0.05,
                    1,
                    1,
                    0.8,
                    0.2,
                    0.05,
                    1,
                    1,
                    0.8,
                    0.2,
                    0.05,
                    1,
                    1,
                    0.8,
                    0.2,
                    0.05,
                    1,
                    1,
                    0.8,
                    0.2,
                    0.05,
                    1,
                    1,
                    0.8,
                    0.2,
                    0.05,
                    1,
                    1,
                    0.8,
                    0.2,
                    0.05,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
            ],
            evidence=["levelglass", "obstacle", "levelbottle", "desirable"],
            evidence_card=[5, 8, 5, 2],
        )

        cdf_str = grasp_cpd._make_table_str(tablefmt="grid")
        terminal_width, terminal_height = get_terminal_size()
        list_rows_str = cdf_str.split("\n")
        table_width, table_length = len(list_rows_str[0]), len(list_rows_str)

        # TODO: test table height

        self.assertGreater(terminal_width, table_width)

    def tearDown(self):
        config.set_backend("numpy")


class TestTabularCPDMethodsTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        sn = {
            "intel": ["low", "medium", "high"],
            "diff": ["low", "high"],
            "grade": [
                "grade(0)",
                "grade(1)",
                "grade(2)",
                "grade(3)",
                "grade(4)",
                "grade(5)",
            ],
        }
        self.cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["intel", "diff"],
            evidence_card=[3, 2],
            state_names=sn,
        )

        self.cpd2 = TabularCPD(
            "J",
            2,
            [
                [0.9, 0.3, 0.9, 0.3, 0.8, 0.8, 0.4, 0.4],
                [0.1, 0.7, 0.1, 0.7, 0.2, 0.2, 0.6, 0.6],
            ],
            evidence=["A", "B", "C"],
            evidence_card=[2, 2, 2],
        )

    def test_marginalize_1(self):
        self.cpd.marginalize(["diff"])
        self.assertEqual(self.cpd.variable, "grade")
        self.assertEqual(self.cpd.variable_card, 3)
        self.assertListEqual(list(self.cpd.variables), ["grade", "intel"])
        np_test.assert_array_equal(self.cpd.cardinality, np.array([3, 3]))
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.cpd.values.ravel()),
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8]),
        )

    def test_marginalize_2(self):
        self.assertRaises(ValueError, self.cpd.marginalize, ["grade"])

    def test_marginalize_3(self):
        copy_cpd = self.cpd.copy()
        copy_cpd.marginalize(["intel", "diff"])
        self.cpd.marginalize(["intel"])
        self.cpd.marginalize(["diff"])
        self.assertTrue(all((self.cpd.values == copy_cpd.values).ravel()))

    def test_normalize(self):
        cpd_un_normalized = TabularCPD(
            "grade",
            2,
            [[0.7, 0.2, 0.6, 0.2], [0.4, 0.4, 0.4, 0.8]],
            ["intel", "diff"],
            [2, 2],
        )
        cpd_un_normalized.normalize()
        np_test.assert_array_almost_equal(
            compat_fns.to_numpy(cpd_un_normalized.values),
            np.array(
                [
                    [[0.63636364, 0.33333333], [0.6, 0.2]],
                    [[0.36363636, 0.66666667], [0.4, 0.8]],
                ]
            ),
        )

    def test_normalize_not_in_place(self):
        cpd_un_normalized = TabularCPD(
            "grade",
            2,
            [[0.7, 0.2, 0.6, 0.2], [0.4, 0.4, 0.4, 0.8]],
            ["intel", "diff"],
            [2, 2],
        )
        np_test.assert_array_almost_equal(
            compat_fns.to_numpy(cpd_un_normalized.normalize(inplace=False).values),
            np.array(
                [
                    [[0.63636364, 0.33333333], [0.6, 0.2]],
                    [[0.36363636, 0.66666667], [0.4, 0.8]],
                ]
            ),
        )

    def test_normalize_original_safe(self):
        cpd_un_normalized = TabularCPD(
            "grade",
            2,
            [[0.7, 0.2, 0.6, 0.2], [0.4, 0.4, 0.4, 0.8]],
            ["intel", "diff"],
            [2, 2],
        )
        cpd_un_normalized.normalize(inplace=False)
        np_test.assert_array_almost_equal(
            compat_fns.to_numpy(cpd_un_normalized.values),
            np.array([[[0.7, 0.2], [0.6, 0.2]], [[0.4, 0.4], [0.4, 0.8]]]),
        )

    def test__repr__(self):
        grade_cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["intel", "diff"],
            evidence_card=[3, 2],
        )
        intel_cpd = TabularCPD("intel", 3, [[0.5], [0.3], [0.2]])
        diff_cpd = TabularCPD(
            "grade",
            3,
            [[0.1, 0.1], [0.1, 0.1], [0.8, 0.8]],
            evidence=["diff"],
            evidence_card=[2],
        )
        self.assertEqual(
            repr(grade_cpd),
            f"<TabularCPD representing P(grade:3 | intel:3, diff:2) at {hex(id(grade_cpd))}>",
        )
        self.assertEqual(
            repr(intel_cpd),
            f"<TabularCPD representing P(intel:3) at {hex(id(intel_cpd))}>",
        )
        self.assertEqual(
            repr(diff_cpd),
            f"<TabularCPD representing P(grade:3 | diff:2) at {hex(id(diff_cpd))}>",
        )

    def test_copy(self):
        copy_cpd = self.cpd.copy()
        self.assertTrue(all((self.cpd.get_values() == copy_cpd.get_values()).ravel()))

    def test_copy_original_safe(self):
        copy_cpd = self.cpd.copy()
        copy_cpd.reorder_parents(["diff", "intel"])
        np_test.assert_almost_equal(
            compat_fns.to_numpy(self.cpd.get_values()),
            np.array(
                [
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                ]
            ),
        )

    def test_copy_state_names(self):
        copy_cpd = self.cpd.copy()
        self.assertEqual(self.cpd.state_names, copy_cpd.state_names)
        copy_cpd.state_names.clear()
        self.assertNotEqual(self.cpd.state_names, copy_cpd.state_names)
        self.assertFalse(copy_cpd.state_names)
        self.assertTrue(self.cpd.state_names)

    def test_reduce_1(self):
        self.cpd.reduce([("diff", "low")])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.cpd.get_values()),
            np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8]]),
        )

    def test_reduce_2(self):
        self.cpd.reduce([("intel", "low")])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.cpd.get_values()),
            np.array([[0.1, 0.1], [0.1, 0.1], [0.8, 0.8]]),
        )

    def test_reduce_3(self):
        self.cpd.reduce([("intel", "low"), ("diff", "low")])
        np_test.assert_array_equal(
            compat_fns.to_numpy(self.cpd.get_values()), np.array([[0.1], [0.1], [0.8]])
        )

    def test_reduce_4(self):
        self.assertRaises(ValueError, self.cpd.reduce, [("grade", "grade(0)")])

    def test_reduce_5(self):
        copy_cpd = self.cpd.copy()
        copy_cpd.reduce([("intel", "high"), ("diff", "low")])
        self.cpd.reduce([("intel", "high")])
        self.cpd.reduce([("diff", "high")])
        self.assertTrue(all((self.cpd.values == copy_cpd.values).ravel()))

    def test_get_values(self):
        np_test.assert_almost_equal(
            compat_fns.to_numpy(self.cpd.get_values()),
            np.array(
                [
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                ]
            ),
        )

    def test_reorder_parents_inplace(self):
        new_vals = self.cpd2.reorder_parents(["B", "A", "C"])
        np_test.assert_almost_equal(
            compat_fns.to_numpy(new_vals),
            np.array(
                [
                    [0.9, 0.3, 0.8, 0.8, 0.9, 0.3, 0.4, 0.4],
                    [0.1, 0.7, 0.2, 0.2, 0.1, 0.7, 0.6, 0.6],
                ]
            ),
        )
        np_test.assert_almost_equal(
            compat_fns.to_numpy(self.cpd2.get_values()),
            np.array(
                [
                    [0.9, 0.3, 0.8, 0.8, 0.9, 0.3, 0.4, 0.4],
                    [0.1, 0.7, 0.2, 0.2, 0.1, 0.7, 0.6, 0.6],
                ]
            ),
        )

    def test_reorder_parents(self):
        new_vals = self.cpd2.reorder_parents(["B", "A", "C"])
        np_test.assert_almost_equal(
            compat_fns.to_numpy(new_vals),
            np.array(
                [
                    [0.9, 0.3, 0.8, 0.8, 0.9, 0.3, 0.4, 0.4],
                    [0.1, 0.7, 0.2, 0.2, 0.1, 0.7, 0.6, 0.6],
                ]
            ),
        )

    def test_reorder_parents_no_effect(self):
        self.cpd2.reorder_parents(["C", "A", "B"], inplace=False)
        np_test.assert_almost_equal(
            compat_fns.to_numpy(self.cpd2.get_values()),
            np.array(
                [
                    [0.9, 0.3, 0.9, 0.3, 0.8, 0.8, 0.4, 0.4],
                    [0.1, 0.7, 0.1, 0.7, 0.2, 0.2, 0.6, 0.6],
                ]
            ),
        )

    def test_reorder_parents_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cpd2.reorder_parents(["A", "B", "C"], inplace=False)
            assert "Same ordering provided as current" in str(w[-1].message)
            np_test.assert_almost_equal(
                compat_fns.to_numpy(self.cpd2.get_values()),
                np.array(
                    [
                        [0.9, 0.3, 0.9, 0.3, 0.8, 0.8, 0.4, 0.4],
                        [0.1, 0.7, 0.1, 0.7, 0.2, 0.2, 0.6, 0.6],
                    ]
                ),
            )

    def test_get_random(self):
        cpd = TabularCPD.get_random(variable="A", evidence=None, cardinality={"A": 3})
        self.assertEqual(cpd.variables, ["A"])
        np_test.assert_array_equal(cpd.cardinality, np.array([3]))
        self.assertEqual(cpd.values.shape, (3,))

        cpd_sn = TabularCPD.get_random(
            variable="A",
            evidence=None,
            cardinality={"A": 3},
            state_names={"A": ["a1", "a2", "a3"]},
        )
        self.assertEqual(cpd_sn.variables, ["A"])
        np_test.assert_array_equal(cpd_sn.cardinality, np.array([3]))
        self.assertEqual(cpd_sn.values.shape, (3,))
        self.assertEqual(cpd_sn.state_names["A"], ["a1", "a2", "a3"])

        cpd = TabularCPD.get_random(
            variable="A", evidence=["B", "C"], cardinality={"A": 2, "B": 3, "C": 4}
        )
        self.assertEqual(cpd.variables, ["A", "B", "C"])
        np_test.assert_array_equal(cpd.cardinality, np.array([2, 3, 4]))
        self.assertEqual(cpd.values.shape, (2, 3, 4))

        cpd_sn = TabularCPD.get_random(
            variable="A",
            evidence=["B", "C"],
            cardinality={"A": 2, "B": 3, "C": 4},
            state_names={
                "A": ["a1", "a2"],
                "B": ["b1", "b2", "b3"],
                "C": ["c1", "c2", "c3", "c4"],
            },
        )
        self.assertEqual(cpd_sn.variables, ["A", "B", "C"])
        np_test.assert_array_equal(cpd_sn.cardinality, np.array([2, 3, 4]))
        self.assertEqual(cpd_sn.values.shape, (2, 3, 4))
        self.assertEqual(cpd_sn.state_names["A"], ["a1", "a2"])
        self.assertEqual(cpd_sn.state_names["B"], ["b1", "b2", "b3"])
        self.assertEqual(cpd_sn.state_names["C"], ["c1", "c2", "c3", "c4"])

        cpd = TabularCPD.get_random(variable="A", evidence=["B", "C"])
        self.assertEqual(cpd.variables, ["A", "B", "C"])
        np_test.assert_array_equal(cpd.cardinality, np.array([2, 2, 2]))
        self.assertEqual(cpd.values.shape, (2, 2, 2))

        cpd = TabularCPD.get_random(
            variable="A",
            evidence=["B", "C"],
            state_names={"A": ["a1", "a2"], "B": ["b1", "b2"], "C": ["c1", "c2"]},
        )
        self.assertEqual(cpd.variables, ["A", "B", "C"])
        np_test.assert_array_equal(cpd.cardinality, np.array([2, 2, 2]))
        self.assertEqual(cpd.values.shape, (2, 2, 2))
        self.assertEqual(cpd.state_names["A"], ["a1", "a2"])
        self.assertEqual(cpd.state_names["B"], ["b1", "b2"])
        self.assertEqual(cpd.state_names["C"], ["c1", "c2"])

        self.assertRaises(
            ValueError,
            TabularCPD.get_random,
            variable="A",
            evidence=["B", "C"],
            cardinality={"A": 2, "B": 3},
        )

    def tearDown(self):
        del self.cpd
        config.set_backend("numpy")


if __name__ == "__main__":
    unittest.main()
