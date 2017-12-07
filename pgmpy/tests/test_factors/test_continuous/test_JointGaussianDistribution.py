import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors.continuous.distributions import GaussianDistribution as BGD
from pgmpy.extern import six


class TestGaussianDistributionInit(unittest.TestCase):
    def test_single_var_joint_dist(self):
        dist_str = BGD(variables='X',
                       mean=1,
                       cov=4)
        np_test.assert_array_equal(dist_str.variables, ['X'])
        np_test.assert_array_equal(dist_str.mean, [[1]])
        np_test.assert_array_equal(dist_str.cov, [[4]])
        np_test.assert_array_equal(dist_str.evidence, [])

        dist_1dlist = BGD(variables=['X'],
                          mean=[1],
                          cov=4)

        np_test.assert_array_equal(dist_1dlist.variables, ['X'])
        np_test.assert_array_equal(dist_1dlist.mean, [[1]])
        np_test.assert_array_equal(dist_1dlist.cov, [[4]])
        np_test.assert_array_equal(dist_1dlist.evidence, [])

        dist_2dlist = BGD(variables=['X'],
                          mean=[[1]],
                          cov=[[4]])

        np_test.assert_array_equal(dist_2dlist.variables, ['X'])
        np_test.assert_array_equal(dist_2dlist.mean, [[1]])
        np_test.assert_array_equal(dist_2dlist.cov, [[4]])
        np_test.assert_array_equal(dist_2dlist.evidence, [])

    def test_multi_var_joint_dist(self):
        dist_1dlist = BGD(variables=['X', 'Y'],
                          mean=[1, 2],
                          cov=[[1, 0], [0, 1]])

        np_test.assert_array_equal(dist_1dlist.variables, ['X', 'Y'])
        np_test.assert_array_equal(dist_1dlist.mean, [[1], [2]])
        np_test.assert_array_equal(dist_1dlist.cov, [[1, 0], [0, 1]])
        np_test.assert_array_equal(dist_1dlist.evidence, [])

        dist_2dlist = BGD(variables=['X', 'Y'],
                          mean=[[1], [2]],
                          cov=[[1, 0], [0, 1]])

        np_test.assert_array_equal(dist_2dlist.variables, ['X', 'Y'])
        np_test.assert_array_equal(dist_2dlist.mean, [[1], [2]])
        np_test.assert_array_equal(dist_2dlist.cov, [[1, 0], [0, 1]])
        np_test.assert_array_equal(dist_2dlist.evidence, [])

        dist_empty_evidence = BGD(variables=['X', 'Y'],
                                  mean=[[1], [2]],
                                  cov=[[1, 0], [0, 1]],
                                  evidence=[])

        np_test.assert_array_equal(dist_empty_evidence.variables, ['X', 'Y'])
        np_test.assert_array_equal(dist_2dlist.mean, [[1], [2]])
        np_test.assert_array_equal(dist_2dlist.cov, [[1, 0], [0, 1]])
        np_test.assert_array_equal(dist_2dlist.evidence, [])

    def test_single_var_cond_dist(self):
        dist_str_non_dependent = BGD(variables='X',
                                     mean=[0, 1],
                                     cov=4,
                                     evidence='A')

        np_test.assert_array_equal(dist_str_non_dependent.variables, ['X'])
        np_test.assert_array_equal(dist_str_non_dependent.mean, [[0, 1]])
        np_test.assert_array_equal(dist_str_non_dependent.cov, [[4]])
        np_test.assert_array_equal(dist_str_non_dependent.evidence, ['A'])

        dist_str_dependent_1dlist = BGD(variables='X',
                                        mean=[2, 0.3],
                                        cov=[[4]],
                                        evidence=['A'])

        np_test.assert_array_equal(dist_str_dependent_1dlist.variables, ['X'])
        np_test.assert_array_equal(dist_str_dependent_1dlist.mean, [[2, 0.3]])
        np_test.assert_array_equal(dist_str_dependent_1dlist.cov, [[4]])
        np_test.assert_array_equal(dist_str_dependent_1dlist.evidence, ['A'])

        dist_str_dependent_2dlist = BGD(variables=['X'],
                                        mean=[[2, 0.3]],
                                        cov=[[4]],
                                        evidence='A')

        np_test.assert_array_equal(dist_str_dependent_2dlist.variables, ['X'])
        np_test.assert_array_equal(dist_str_dependent_2dlist.mean, [[2, 0.3]])
        np_test.assert_array_equal(dist_str_dependent_2dlist.cov, [[4]])
        np_test.assert_array_equal(dist_str_dependent_2dlist.evidence, ['A'])

    def test_multi_var_cond_dist(self):
        dist_single_evidence_list = BGD(variables=['X', 'Y'],
                                        mean=[[2, 0.3], [1, 0.5]],
                                        cov=[[0.3, 0.5], [0.5, 0.3]],
                                        evidence=['A'])

        np_test.assert_array_equal(dist_single_evidence_list.variables, ['X', 'Y'])
        np_test.assert_array_equal(dist_single_evidence_list.mean, [[2, 0.3], [1, 0.5]])
        np_test.assert_array_equal(dist_single_evidence_list.cov, [[0.3, 0.5], [0.5, 0.3]])
        np_test.assert_array_equal(dist_single_evidence_list.evidence, ['A'])

        dist_single_evidence_str = BGD(variables=['X', 'Y'],
                                       mean=[[2, 0.3], [1, 0.5]],
                                       cov=[[0.3, 0.5], [0.5, 0.3]],
                                       evidence='A')

        np_test.assert_array_equal(dist_single_evidence_str.variables, ['X', 'Y'])
        np_test.assert_array_equal(dist_single_evidence_str.mean, [[2, 0.3], [1, 0.5]])
        np_test.assert_array_equal(dist_single_evidence_str.cov, [[0.3, 0.5], [0.5, 0.3]])
        np_test.assert_array_equal(dist_single_evidence_str.evidence, ['A'])

        dist_multi_evidence_single_var_1dlist = BGD(variables='X',
                                                    mean=[1, 0.4, 0.5],
                                                    cov=4,
                                                    evidence=['A', 'B'])

        np_test.assert_array_equal(dist_multi_evidence_single_var_1dlist.variables, ['X'])
        np_test.assert_array_equal(dist_multi_evidence_single_var_1dlist.mean, [[1, 0.4, 0.5]])
        np_test.assert_array_equal(dist_multi_evidence_single_var_1dlist.cov, [[4]])
        np_test.assert_array_equal(dist_multi_evidence_single_var_1dlist.evidence, ['A', 'B'])

        dist_multi_evidence_single_var_2dlist = BGD(variables='X',
                                                    mean=[[1, 0.4, 0.5]],
                                                    cov=4,
                                                    evidence=['A', 'B'])

        np_test.assert_array_equal(dist_multi_evidence_single_var_2dlist.variables, ['X'])
        np_test.assert_array_equal(dist_multi_evidence_single_var_2dlist.mean, [[1, 0.4, 0.5]])
        np_test.assert_array_equal(dist_multi_evidence_single_var_2dlist.cov, [[4]])
        np_test.assert_array_equal(dist_multi_evidence_single_var_2dlist.evidence, ['A', 'B'])

        dist_multi_evid_multi_var = BGD(variables=['X', 'Y'],
                                        mean=[[1, 0.4, 0.6],
                                              [0.5, 3, 0.6]],
                                        cov=[[0.5, 0.2],
                                             [0.2, 0.8]],
                                        evidence=['A', 'B'])

        np_test.assert_array_equal(dist_multi_evid_multi_var.variables, ['X', 'Y'])
        np_test.assert_array_equal(dist_multi_evid_multi_var.mean, [[1, 0.4, 0.6],
                                                                    [0.5, 3, 0.6]])
        np_test.assert_array_equal(dist_multi_evid_multi_var.cov, [[0.5, 0.2], [0.2, 0.8]])
        np_test.assert_array_equal(dist_multi_evid_multi_var.evidence, ['A', 'B'])


class TestGaussianDistributionMethods(unittest.TestCase):
    """
    All values in the tests have been generated from scipy.stats.multivariate_normal
    """
    def setUp(self):
        self.joint_dist = BGD(variables=['X', 'Y', 'Z'], mean=[1, 1, 1],
                              cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.cond_dist = BGD(variables=['X', 'Y', 'Z'],
                             mean=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                             cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                             evidence=['A', 'B'])

        self.joint_dist_c = BGD(variables=['X', 'Y', 'Z'], mean=[0.2, 0.3, 0.4],
                                cov=[[0.4, 0.8, 0.7], [0.8, 0.9, 0.5], [0.7, 0.5, 1]])
        self.cond_dist_c = BGD(variables=['X', 'Y', 'Z'],
                               mean=[[0.4, 0.8, 0.2], [0.7, 0.9, 0.4], [0.2, 0.3, 0.7]],
                               cov=[[0.4, 0.8, 0.7], [0.8, 0.9, 0.5], [0.7, 0.5, 1]],
                               evidence=['A', 'B'])
        self.points = [[-0.34098948,  1.56544901,  0.44807747],
                       [ 0.35044270, -0.43818711, -0.73991183],
                       [-0.29182544,  0.96646830, -0.13819649]]

    def test_pdf_single_point(self):
        pdf_joint = self.joint_dist.pdf()
        self.assertEqual(pdf_joint([1, 1, 1]), 0.063493635934240983)
        self.assertEqual(pdf_joint([-1, 0, 1]), 0.0052118750182884995)

        pdf_cond = self.cond_dist.pdf(evidence_values=[0, 0])
        self.assertEqual(pdf_cond([1, 1, 1]), 0.063493635934240983)
        self.assertEqual(pdf_cond([-1, 0, 1]), 0.0052118750182884995)

        pdf_cond = self.cond_dist.pdf(evidence_values=[1, 1])
        self.assertEqual(pdf_cond([1, 1, 1]), 0.00015738498827646107)
        self.assertEqual(pdf_cond([-1, 0, 1]), 3.2022866871370889e-08)

    def test_pdf_multi_point(self):
        pdf_joint = self.joint_dist.pdf()
        np_test.assert_almost_equal(
            pdf_joint(self.points), np.array([0.018909, 0.00402345, 0.01441437]))

        pdf_cond = self.cond_dist.pdf(evidence_values=[0, 0])
        np_test.assert_almost_equal(pdf_cond(self.points),
                                    np.array([0.018909, 0.00402345, 0.01441437]))

        pdf_cond = self.cond_dist.pdf(evidence_values=[1, 1])
        np_test.assert_almost_equal(
            pdf_cond(self.points),
            np.array([3.29520049e-06, 4.72249715e-09, 2.58948364e-07]))

    def test_assignment(self):
        self.assertEqual(self.joint_dist.assignment([1, 1, 1]), 0.063493635934240983)
        self.assertEqual(self.joint_dist.assignment([-1, 0, 1]), 0.0052118750182884995)

        self.assertEqual(self.cond_dist.assignment(
            x=[1, 1, 1], evidence_values=[0, 0]), 0.063493635934240983)
        self.assertEqual(self.cond_dist.assignment(
            x=[-1, 0, 1], evidence_values=[0, 0]), 0.0052118750182884995)

        np_test.assert_almost_equal(self.joint_dist.assignment(x=self.points),
                                    np.array([0.018909, 0.00402345, 0.01441437]))
        np_test.assert_almost_equal(self.cond_dist.assignment(x=self.points,
                                                              evidence_values=[0, 0]),
                                    np.array([0.018909, 0.00402345, 0.01441437]))

    def test_copy(self):
        copy_joint = self.joint_dist.copy()
        copy_joint.variables[1] = 'M'
        copy_joint.mean[1] = 2
        copy_joint.cov[1, 1] = 2

        np_test.assert_equal(self.joint_dist.variables, ['X', 'Y', 'Z'])
        np_test.assert_equal(copy_joint.variables, ['X', 'M', 'Z'])
        np_test.assert_equal(self.joint_dist.mean, [[1], [1], [1]])
        np_test.assert_equal(copy_joint.mean, [[1], [2], [1]])
        np_test.assert_equal(self.joint_dist.cov, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np_test.assert_equal(copy_joint.cov, [[1, 0, 0], [0, 2, 0], [0, 0, 1]])

        copy_cond = self.cond_dist.copy()
        copy_cond.variables[1] = 'M'
        copy_cond.mean[1, 1] = 2
        copy_cond.cov[1, 1] = 2
        copy_cond.evidence[1] = 'C'

        np_test.assert_equal(self.cond_dist.variables, ['X', 'Y', 'Z'])
        np_test.assert_equal(copy_cond.variables, ['X', 'M', 'Z'])
        np_test.assert_equal(self.cond_dist.mean, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        np_test.assert_equal(copy_cond.mean, [[1, 1, 1], [1, 2, 1], [1, 1, 1]])
        np_test.assert_equal(self.cond_dist.cov, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np_test.assert_equal(copy_cond.cov, [[1, 0, 0], [0, 2, 0], [0, 0, 1]])

    def test_str(self):
        self.assertEqual(self.joint_dist.__str__(),
                         "P(X, Y, Z) = N(1, 1, 1; [1 0 0], [0 1 0], [0 0 1])")
        self.assertEqual(self.cond_dist.__str__(),
                         "P(X, Y, Z | A, B) = N(1A + 1B + 1, 1A + 1B + 1, 1A + 1B + 1; " +
                         "[1 0 0], [0 1 0], [0 0 1])")

    def test_repr(self):
        self.assertEqual(self.joint_dist.__repr__(),
                         "<GaussianDistribution representing P(X, Y, Z) at " +
                         str(hex(id(self.joint_dist))) + ">")
        self.assertEqual(self.cond_dist.__repr__(),
                         "<GaussianDistribution representing P(X, Y, Z | A, B) at " +
                         str(hex(id(self.cond_dist))) + ">")

    def test_reorder_vars(self):
        # Reorder variables in Joint Distribution
        reordered_vars_dist = self.joint_dist_c._reorder_vars(variables=['Y', 'Z', 'X'],
                                                              inplace=False)
        self.assertEqual(list(reordered_vars_dist.variables), ['Y', 'Z', 'X'])
        np_test.assert_almost_equal(reordered_vars_dist.mean, [[0.3], [0.4], [0.2]])
        np_test.assert_almost_equal(reordered_vars_dist.cov, [[0.9, 0.5, 0.8],
                                                              [0.5, 1.0, 0.7],
                                                              [0.8, 0.7, 0.4]])
        self.assertEqual(list(reordered_vars_dist.evidence), [])

        # Reorder just variables in Conditional Distribution
        reordered_vars_dist = self.cond_dist_c._reorder_vars(variables=['Y', 'Z', 'X'],
                                                             inplace=False)
        self.assertEqual(list(reordered_vars_dist.variables), ['Y', 'Z', 'X'])
        np_test.assert_almost_equal(reordered_vars_dist.mean, [[0.7, 0.9, 0.4],
                                                               [0.2, 0.3, 0.7],
                                                               [0.4, 0.8, 0.2]])
        np_test.assert_almost_equal(reordered_vars_dist.cov, [[0.9, 0.5, 0.8],
                                                              [0.5, 1.0, 0.7],
                                                              [0.8, 0.7, 0.4]])
        self.assertEqual(list(reordered_vars_dist.evidence), ['A', 'B'])

        # Reorder just evidence in conditional distribution.
        reordered_evi_dist = self.cond_dist_c._reorder_vars(evidence=['B', 'A'],
                                                            inplace=False)
        self.assertEqual(list(reordered_evi_dist.variables), ['X', 'Y', 'Z'])
        np_test.assert_almost_equal(reordered_evi_dist.mean, [[0.8, 0.4, 0.2],
                                                              [0.9, 0.7, 0.4],
                                                              [0.3, 0.2, 0.7]])
        np_test.assert_almost_equal(reordered_evi_dist.cov, [[0.4, 0.8, 0.7],
                                                             [0.8, 0.9, 0.5],
                                                             [0.7, 0.5, 1.0]])
        self.assertEqual(list(reordered_evi_dist.evidence), ['B', 'A'])

        # Reorder both variables and evidence in conditional distribution.
        reordered_var_evi_dist = self.cond_dist_c._reorder_vars(variables=['Y', 'Z', 'X'],
                                                                evidence=['B', 'A'],
                                                                inplace=False)
        self.assertEqual(list(reordered_var_evi_dist.variables), ['Y', 'Z', 'X'])
        np_test.assert_array_equal(reordered_var_evi_dist.mean, [[0.9, 0.7, 0.4],
                                                                 [0.3, 0.2, 0.7],
                                                                 [0.8, 0.4, 0.2]])
        np_test.assert_almost_equal(reordered_var_evi_dist.cov, [[0.9, 0.5, 0.8],
                                                                 [0.5, 1.0, 0.7],
                                                                 [0.8, 0.7, 0.4]])
        self.assertEqual(list(reordered_var_evi_dist.evidence), ['B', 'A'])

        # Reorder both variables and evidence in conditional distribution using
        # numpy array as an argument.
        reordered_var_evi_dist = self.cond_dist_c._reorder_vars(
            variables=np.array(['Y', 'Z', 'X']),
            evidence=np.array(['B', 'A']),
            inplace=False)

        self.assertEqual(list(reordered_var_evi_dist.variables), ['Y', 'Z', 'X'])
        np_test.assert_array_equal(reordered_var_evi_dist.mean, [[0.9, 0.7, 0.4],
                                                                 [0.3, 0.2, 0.7],
                                                                 [0.8, 0.4, 0.2]])
        np_test.assert_almost_equal(reordered_var_evi_dist.cov, [[0.9, 0.5, 0.8],
                                                                 [0.5, 1.0, 0.7],
                                                                 [0.8, 0.7, 0.4]])
        self.assertEqual(list(reordered_var_evi_dist.evidence), ['B', 'A'])

    def test_reorder_vars_errors(self):
        self.assertRaises(ValueError, self.joint_dist._reorder_vars, variables=['Y', 'X'])
        self.assertRaises(ValueError, self.cond_dist._reorder_vars, variables=['Y'])
        self.assertRaises(ValueError, self.joint_dist._reorder_vars, evidence=['A'])
        self.assertRaises(ValueError, self.cond_dist._reorder_vars, evidence=['B'])
        self.assertRaises(ValueError, self.joint_dist._reorder_vars,
                          variables=['Y', 'X'], evidence=['B'])
        self.assertRaises(ValueError, self.cond_dist._reorder_vars,
                          variables=['Y', 'X'], evidence=['B'])

    def test_marginalize(self):
        marginal_joint = self.joint_dist.marginalize(['Y'], inplace=False)
        self.assertEqual(list(marginal_joint.variables), ['X', 'Z'])
        self.assertEqual(list(marginal_joint.evidence), [])
        np_test.assert_almost_equal(marginal_joint.mean, [[1], [1]])
        np_test.assert_array_almost_equal(marginal_joint.cov, [[1, 0], [0, 1]])

        marginal_cond = self.cond_dist.marginalize(['Y'], inplace=False)
        self.assertEqual(list(marginal_cond.variables), ['X', 'Z'])
        self.assertEqual(list(marginal_cond.evidence), ['A', 'B'])
        np_test.assert_almost_equal(marginal_cond.mean, [[1, 1, 1], [1, 1, 1]])
        np_test.assert_almost_equal(marginal_cond.cov, [[1, 0], [0, 1]])

        marginal_joint_c = self.joint_dist_c.marginalize(['X', 'Z'], inplace=False)
        self.assertEqual(list(marginal_joint_c.variables), ['Y'])
        self.assertEqual(list(marginal_joint_c.evidence), [])
        np_test.assert_almost_equal(marginal_joint_c.mean, [[0.3]])
        np_test.assert_almost_equal(marginal_joint_c.cov, [[0.9]])

        marginal_cond_c = self.cond_dist_c.marginalize(['Z'], inplace=False)
        self.assertEqual(list(marginal_cond_c.variables), ['X', 'Y'])
        self.assertEqual(list(marginal_cond_c.evidence), ['A', 'B'])
        np_test.assert_almost_equal(marginal_cond_c.mean, [[0.4, 0.8, 0.2], [0.7, 0.9, 0.4]])
        np_test.assert_almost_equal(marginal_cond_c.cov, [[0.4, 0.8], [0.8, 0.9]])

        marginal_cond_c = self.cond_dist_c.marginalize('Z', inplace=False)
        self.assertEqual(list(marginal_cond_c.variables), ['X', 'Y'])
        self.assertEqual(list(marginal_cond_c.evidence), ['A', 'B'])
        np_test.assert_almost_equal(marginal_cond_c.mean, [[0.4, 0.8, 0.2], [0.7, 0.9, 0.4]])
        np_test.assert_almost_equal(marginal_cond_c.cov, [[0.4, 0.8], [0.8, 0.9]])

    @unittest.skipIf(six.PY2, "Skipping for python 2 because using assertWarns")
    def test_marginalize_warning(self):
        with self.assertWarns(Warning):
            self.joint_dist.marginalize(['X', 'A'], inplace=False)

        with self.assertWarns(Warning):
            self.joint_dist.marginalize('A', inplace=False)

        with self.assertWarns(Warning):
            self.cond_dist.marginalize(['A', 'Y'], inplace=False)

        with self.assertWarns(Warning):
            self.cond_dist_c.marginalize('A', inplace=False)

        with self.assertWarns(Warning):
            self.cond_dist.marginalize(['X', 'X'], inplace=False)

        with self.assertWarns(Warning):
            self.joint_dist.marginalize(['A', 'A'], inplace=False)

    def test_reduce_with_values(self):
        reduced_joint = self.joint_dist.reduce(variables=[('Y', 1)], inplace=False)
        self.assertEqual(list(reduced_joint.variables), ['X', 'Z'])
        np_test.assert_almost_equal(reduced_joint.mean, [[1], [1]], decimal=2)
        np_test.assert_almost_equal(reduced_joint.cov, [[1, 0], [0, 1]], decimal=2)
        np_test.assert_almost_equal(reduced_joint.evidence, [])

        reduced_joint_c = self.joint_dist_c.reduce(
            variables=[('Y', 0.5), ('Z', 0.8)], inplace=False)
        self.assertEqual(list(reduced_joint_c.variables), ['X'])
        np_test.assert_almost_equal(reduced_joint_c.mean, [[0.48]], decimal=2)
        np_test.assert_almost_equal(reduced_joint_c.cov, [[-0.4]], decimal=2)
        np_test.assert_almost_equal(reduced_joint_c.evidence, [])

        reduced_cond = self.cond_dist.reduce(variables=[('Y', 1)], inplace=False)
        self.assertEqual(list(reduced_cond.variables), ['X', 'Z'])
        np_test.assert_almost_equal(reduced_cond.mean, [[1, 1, 1], [1, 1, 1]])
        np_test.assert_almost_equal(reduced_cond.cov, [[1, 0], [0, 1]])
        self.assertEqual(list(reduced_cond.evidence), ['A', 'B'])

        reduced_cond_c = self.cond_dist_c.reduce(variables=[('X', 0.4), ('Y', 0.8)],
                                                 inplace=False)
        self.assertEqual(list(reduced_cond_c.variables), ['Z'])
        np_test.assert_almost_equal(reduced_cond_c.mean, [[0.7714286, 0.8, 1.05]],
                                    decimal=4)
        np_test.assert_almost_equal(reduced_cond_c.cov, [[0.932142]], decimal=4)
        self.assertEqual(list(reduced_cond_c.evidence), ['A', 'B'])

    def test_reduce_without_values(self):
        reduced_joint = self.joint_dist.reduce(variables=['Y'], inplace=False)
        self.assertEqual(list(reduced_joint.variables), ['X', 'Z'])
        np_test.assert_almost_equal(reduced_joint.mean, [[0, 1], [0, 1]])
        np_test.assert_almost_equal(reduced_joint.cov, [[1, 0], [0, 1]])
        self.assertEqual(list(reduced_joint.evidence), ['Y'])

        reduced_joint_c = self.joint_dist_c.reduce(variables=['X', 'Z'],
                                                   inplace=False)
        self.assertEqual(list(reduced_joint_c.variables), ['Y'])
        np_test.assert_almost_equal(reduced_joint_c.mean, [[-5.0, 4.0, -0.3]],
                                    decimal=4)
        np_test.assert_almost_equal(reduced_joint_c.cov, [[2.9]],
                                    decimal=4)
        self.assertEqual(list(reduced_joint_c.evidence), ['X', 'Z'])

        reduced_cond = self.cond_dist.reduce(variables=['Y'], inplace=False)
        self.assertEqual(list(reduced_cond.variables), ['X', 'Z'])
        np_test.assert_almost_equal(reduced_cond.mean, [[0, 1, 1, 1], [0, 1, 1, 1]])
        np_test.assert_almost_equal(reduced_cond.cov, [[1, 0], [0, 1]])
        self.assertEqual(list(reduced_cond.evidence), ['Y', 'A', 'B'])

        reduced_cond_c = self.cond_dist_c.reduce(variables=['X', 'Z'], inplace=False)
        self.assertEqual(list(reduced_cond_c.variables), ['Y'])
        np_test.assert_almost_equal(reduced_cond_c.mean, [[-5, 4, 1.9, 3.7, -1.4]],
                                    decimal=4)
        np_test.assert_almost_equal(reduced_cond_c.cov, [[2.9]],
                                    decimal=4)
        self.assertEqual(list(reduced_cond_c.evidence), ['X', 'Z', 'A', 'B'])

    def test_add_independent_variables(self):
        dist_new = self.joint_dist._add_independent_variables(variables=[('A', 1, 1)], inplace=False)
        self.assertEqual(list(dist_new.variables), ['X', 'Y', 'Z', 'A'])
        np_test.assert_almost_equal(dist_new.mean, [[1], [1], [1], [1]])
        np_test.assert_almost_equal(dist_new.cov, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertEqual(list(dist_new.evidence), [])

        dist_new = self.joint_dist._add_independent_variables(evidence=['C', 'D'], inplace=False)
        self.assertEqual(list(dist_new.variables), ['X', 'Y', 'Z'])
        np_test.assert_almost_equal(dist_new.mean, [[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        np_test.assert_almost_equal(dist_new.cov, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(list(dist_new.evidence), ['C', 'D'])

        dist_new = self.joint_dist._add_independent_variables(variables=[('L', 1, 1), ('M', 2, 2)],
                                                              evidence=['C', 'D'], inplace=False)
        self.assertEqual(list(dist_new.variables), ['X', 'Y', 'Z', 'L', 'M'])
        np_test.assert_almost_equal(dist_new.mean, [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 2]])
        np_test.assert_almost_equal(dist_new.cov, [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                                                   [0, 0, 0, 1, 0], [0, 0, 0, 0, 2]])
        self.assertEqual(list(dist_new.evidence), ['C', 'D'])

        dist_new = self.cond_dist._add_independent_variables(variables=[('L', 1, 1)], inplace=False)
        self.assertEqual(list(dist_new.variables), ['X', 'Y', 'Z', 'L'])
        np_test.assert_almost_equal(dist_new.mean, [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1]])
        np_test.assert_almost_equal(dist_new.cov, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertEqual(list(dist_new.evidence), ['A', 'B'])

        dist_new = self.cond_dist._add_independent_variables(evidence=['C', 'D'], inplace=False)
        self.assertEqual(list(dist_new.variables), ['X', 'Y', 'Z'])
        np_test.assert_almost_equal(dist_new.mean, [[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1]])
        np_test.assert_almost_equal(dist_new.cov, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(list(dist_new.evidence), ['C', 'D', 'A', 'B'])

        dist_new = self.cond_dist._add_independent_variables(variables=[('L', 1, 1), ('M', 2, 2)],
                                                             evidence=['C', 'D'], inplace=False)
        self.assertEqual(list(dist_new.variables), ['X', 'Y', 'Z', 'L', 'M'])
        np_test.assert_almost_equal(dist_new.mean, [[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1],
                                                    [0, 0, 0, 0, 1], [0, 0, 0, 0, 2]])
        np_test.assert_almost_equal(dist_new.cov, [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                                                   [0, 0, 0, 1, 0], [0, 0, 0, 0, 2]])
        self.assertEqual(list(dist_new.evidence), ['C', 'D', 'A', 'B'])

    def tearDown(self):
        del self.joint_dist
        del self.cond_dist
