import unittest
import numpy as np
from pgmpy.models import MarkovModel
from pgmpy.inference.mplp import Mplp
from pgmpy.factors import Factor


class TestMplp(unittest.TestCase):
    def setUp(self):

        self.markov_model = MarkovModel()

        self.markov_model.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('E', 'F'),
                                ('F', 'G'), ('G', 'H'), ('D', 'H'), ('H', 'L'),
                                ('I', 'J'), ('J', 'K'), ('K', 'L'), ('L', 'P'),
                                ('M', 'N'), ('N', 'O'), ('O', 'P'), ('A', 'E'),
                                ('E', 'I'), ('I', 'M'), ('B', 'F'), ('F', 'J'),
                                ('J', 'N'), ('C', 'G'), ('G', 'K'), ('K', 'O')])

        factor_a = Factor(['A'], cardinality=[2], value=np.array([0.54577, 1.8323]))
        factor_b = Factor(['B'], cardinality=[2], value=np.array([0.93894, 1.065]))
        factor_c = Factor(['C'], cardinality=[2], value=np.array([0.89205, 1.121]))
        factor_d = Factor(['D'], cardinality=[2], value=np.array([0.56292, 1.7765]))
        factor_e = Factor(['E'], cardinality=[2], value=np.array([0.47117, 2.1224]))
        factor_f = Factor(['F'], cardinality=[2], value=np.array([1.5093, 0.66257]))
        factor_g = Factor(['G'], cardinality=[2], value=np.array([0.48011, 2.0828]))
        factor_h = Factor(['H'], cardinality=[2], value=np.array([2.6332, 0.37977]))
        factor_i = Factor(['I'], cardinality=[2], value=np.array([1.992, 0.50202]))
        factor_j = Factor(['J'], cardinality=[2], value=np.array([1.6443, 0.60817]))
        factor_k = Factor(['K'], cardinality=[2], value=np.array([0.39909, 2.5057]))
        factor_l = Factor(['L'], cardinality=[2], value=np.array([1.9965, 0.50087]))
        factor_m = Factor(['M'], cardinality=[2], value=np.array([2.4581, 0.40681]))
        factor_n = Factor(['N'], cardinality=[2], value=np.array([2.0481, 0.48826]))
        factor_o = Factor(['O'], cardinality=[2], value=np.array([0.6477, 1.5439]))
        factor_p = Factor(['P'], cardinality=[2], value=np.array([0.93844, 1.0656]))
        factor_a_b = Factor(['A', 'B'], cardinality=[2, 2], value=np.array([1.3207, 0.75717, 0.75717, 1.3207]))
        factor_b_c = Factor(['B', 'C'], cardinality=[2, 2], value=np.array([0.00024189, 4134.2, 4134.2, 0.00024189]))
        factor_c_d = Factor(['C', 'D'], cardinality=[2, 2], value=np.array([0.0043227, 231.34, 231.34, 0.0043227]))
        factor_e_f = Factor(['E', 'F'], cardinality=[2, 2], value=np.array([31.228, 0.032023, 0.032023, 31.228]))
        factor_f_g = Factor(['F', 'G'], cardinality=[2, 2], value=np.array([0.43897, 2.278, 2.278, 0.43897]))
        factor_g_h = Factor(['G', 'H'], cardinality=[2, 2], value=np.array([3033.9, 0.00032961, 0.00032961, 3033.9]))
        factor_i_j = Factor(['I', 'J'], cardinality=[2, 2], value=np.array([314.11, 0.0031836, 0.0031836, 314.11]))
        factor_j_k = Factor(['J', 'K'], cardinality=[2, 2], value=np.array([0.3764, 2.6568, 2.6568, 0.3764]))
        factor_k_l = Factor(['K', 'L'], cardinality=[2, 2], value=np.array([3892.6, 0.0002569, 0.0002569, 3892.6]))
        factor_m_n = Factor(['M', 'N'], cardinality=[2, 2], value=np.array([0.014559, 68.687, 68.687, 0.014559]))
        factor_n_o = Factor(['N', 'O'], cardinality=[2, 2], value=np.array([20.155, 0.049615, 0.049615, 20.155]))
        factor_o_p = Factor(['O', 'P'], cardinality=[2, 2], value=np.array([0.013435, 74.435, 74.435, 0.013435]))
        factor_a_e = Factor(['A', 'E'], cardinality=[2, 2], value=np.array([16.14, 0.061959, 0.061959, 16.14]))
        factor_e_i = Factor(['E', 'I'], cardinality=[2, 2], value=np.array([0.001312, 762.21, 762.21, 0.001312]))
        factor_i_m = Factor(['I', 'M'], cardinality=[2, 2], value=np.array([0.00099042, 1009.7, 1009.7, 0.00099042]))
        factor_b_f = Factor(['B', 'F'], cardinality=[2, 2], value=np.array([352.33, 0.0028383, 0.0028383, 352.33]))
        factor_f_j = Factor(['F', 'J'], cardinality=[2, 2], value=np.array([19.263, 0.051912, 0.051912, 19.263]))
        factor_j_n = Factor(['J', 'N'], cardinality=[2, 2], value=np.array([141.3, 0.007077, 0.007077, 141.3]))
        factor_c_g = Factor(['C', 'G'], cardinality=[2, 2], value=np.array([0.00023442, 4265.9, 4265.9, 0.00023442]))
        factor_g_k = Factor(['G', 'K'], cardinality=[2, 2], value=np.array([134.43, 0.0074387, 0.0074387, 134.43]))
        factor_k_o = Factor(['K', 'O'], cardinality=[2, 2], value=np.array([0.00015823, 6320.0, 6320.0, 0.00015823]))
        factor_d_h = Factor(['D', 'H'], cardinality=[2, 2], value=np.array([1994.0, 0.0005015, 0.0005015, 1994.0]))
        factor_h_l = Factor(['H', 'L'], cardinality=[2, 2], value=np.array([0.022576, 44.295, 44.295, 0.022576]))
        factor_l_p = Factor(['L', 'P'], cardinality=[2, 2], value=np.array([0.0018291, 546.72, 546.72, 0.0018291]))

        self.markov_model.add_factors(factor_a, factor_b, factor_c, factor_d,
                                      factor_e, factor_f, factor_g, factor_h,
                                      factor_i, factor_j, factor_k, factor_l,
                                      factor_m, factor_n, factor_o, factor_p,
                                      factor_a_b, factor_b_c, factor_c_d, factor_e_f,
                                      factor_f_g, factor_g_h, factor_i_j, factor_j_k,
                                      factor_k_l, factor_m_n, factor_n_o, factor_o_p,
                                      factor_a_e, factor_e_i, factor_i_m, factor_b_f,
                                      factor_f_j, factor_j_n, factor_c_g, factor_g_k,
                                      factor_k_o, factor_d_h, factor_h_l, factor_l_p)

        for factor in self.markov_model.factors:
            factor.values = np.log(factor.values)
        self.mplp = Mplp(self.markov_model)

    def test_query_single_variable(self):
        query_result = self.mplp.map_query(1000, 0.0002, 0.0002)
        # Results from the Sontag code for a mplp run without tightening is:
        expected_result = {
            'A': 0.60557, 'B': -0.06300, 'C': 0.11422, 'D': -0.57461,
            'E': 0.75254, 'F': 0.41164, 'G': 0.73371, 'H': 0.96819,
            'I': 0.68913, 'J': 0.49731, 'K': 0.91856, 'L': -0.69140,
            'M': -0.89940, 'N': 0.71691, 'O': -0.43432, 'P': -0.06353}

        self.assertAlmostEqual(expected_result['A'], query_result['A'], places=4)
        self.assertAlmostEqual(expected_result['B'], query_result['B'], places=4)
        self.assertAlmostEqual(expected_result['C'], query_result['C'], places=4)
        self.assertAlmostEqual(expected_result['D'], query_result['D'], places=4)
        self.assertAlmostEqual(expected_result['E'], query_result['E'], places=4)
        self.assertAlmostEqual(expected_result['F'], query_result['F'], places=4)
        self.assertAlmostEqual(expected_result['G'], query_result['G'], places=4)
        self.assertAlmostEqual(expected_result['H'], query_result['H'], places=4)
        self.assertAlmostEqual(expected_result['I'], query_result['I'], places=4)
        self.assertAlmostEqual(expected_result['J'], query_result['J'], places=4)
        self.assertAlmostEqual(expected_result['K'], query_result['K'], places=4)
        self.assertAlmostEqual(expected_result['L'], query_result['L'], places=4)
        self.assertAlmostEqual(expected_result['M'], query_result['M'], places=4)
        self.assertAlmostEqual(expected_result['N'], query_result['N'], places=4)
        self.assertAlmostEqual(expected_result['O'], query_result['O'], places=4)
        self.assertAlmostEqual(expected_result['P'], query_result['P'], places=4)