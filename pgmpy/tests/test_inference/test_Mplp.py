import unittest

import numpy as np

from pgmpy.inference.mplp import Mplp
from pgmpy.readwrite import UAIReader


class TestMplp(unittest.TestCase):
    def setUp(self):

        reader_file = UAIReader('pgmpy/tests/test_readwrite/testdata/grid4x4_with_triplets.uai')
        self.markov_model = reader_file.get_model()

        for factor in self.markov_model.factors:
            factor.values = np.log(factor.values)
        self.mplp = Mplp(self.markov_model)


class TightenTripletOff(TestMplp):

    # Query when tighten triplet is OFF
    def test_query_tighten_triplet_off(self):
        query_result = self.mplp.map_query(tighten_triplet=False)

        # Results from the Sontag code for a mplp run without tightening is:
        expected_result = {
            'P': -0.06353, 'N': 0.71691, 'O': 0.43431, 'L': -0.69140,
            'M': -0.89940, 'J': 0.49731, 'K': 0.91856, 'H': 0.96819,
            'I': 0.68913, 'F': 0.41164, 'G': 0.73371, 'D': -0.57461,
            'E': 0.75254, 'B': 0.06297, 'C': -0.11423, 'A': 0.60557}

        self.assertAlmostEqual(expected_result['A'], query_result['var_0'], places=4)
        self.assertAlmostEqual(expected_result['B'], query_result['var_1'], places=4)
        self.assertAlmostEqual(expected_result['C'], query_result['var_2'], places=4)
        self.assertAlmostEqual(expected_result['D'], query_result['var_3'], places=4)
        self.assertAlmostEqual(expected_result['E'], query_result['var_4'], places=4)
        self.assertAlmostEqual(expected_result['F'], query_result['var_5'], places=4)
        self.assertAlmostEqual(expected_result['G'], query_result['var_6'], places=4)
        self.assertAlmostEqual(expected_result['H'], query_result['var_7'], places=4)
        self.assertAlmostEqual(expected_result['I'], query_result['var_8'], places=4)
        self.assertAlmostEqual(expected_result['J'], query_result['var_9'], places=4)
        self.assertAlmostEqual(expected_result['K'], query_result['var_10'], places=4)
        self.assertAlmostEqual(expected_result['L'], query_result['var_11'], places=4)
        self.assertAlmostEqual(expected_result['M'], query_result['var_12'], places=4)
        self.assertAlmostEqual(expected_result['N'], query_result['var_13'], places=4)
        self.assertAlmostEqual(expected_result['O'], query_result['var_14'], places=4)
        self.assertAlmostEqual(expected_result['P'], query_result['var_15'], places=4)

        # The final Integrality gap after solving for the present case
        int_gap = self.mplp.get_integrality_gap()
        self.assertAlmostEqual(64.59, int_gap, places=1)


class TightenTripletOn(TestMplp):

    # Query when tighten triplet is ON
    def test_query_tighten_triplet_on(self):
        query_result = self.mplp.map_query(tighten_triplet=True)
        # Results from the Sontag code for a mplp run with tightening is:
        expected_result = {
            'P': 0.06353, 'C': 0.11422, 'B': -0.06300, 'A': 0.60557,
            'G': -0.73374, 'F': 0.41164, 'E': 0.75254, 'D': -0.57461,
            'K': -0.91856, 'J': 0.49731, 'I': 0.68913, 'H': 0.96819,
            'O': 0.43431, 'N': 0.71691, 'M': -0.89940, 'L': 0.69139}

        self.assertAlmostEqual(expected_result['A'], query_result['var_0'], places=4)
        self.assertAlmostEqual(expected_result['B'], query_result['var_1'], places=4)
        self.assertAlmostEqual(expected_result['C'], query_result['var_2'], places=4)
        self.assertAlmostEqual(expected_result['D'], query_result['var_3'], places=4)
        self.assertAlmostEqual(expected_result['E'], query_result['var_4'], places=4)
        self.assertAlmostEqual(expected_result['F'], query_result['var_5'], places=4)
        self.assertAlmostEqual(expected_result['G'], query_result['var_6'], places=4)
        self.assertAlmostEqual(expected_result['H'], query_result['var_7'], places=4)
        self.assertAlmostEqual(expected_result['I'], query_result['var_8'], places=4)
        self.assertAlmostEqual(expected_result['J'], query_result['var_9'], places=4)
        self.assertAlmostEqual(expected_result['K'], query_result['var_10'], places=4)
        self.assertAlmostEqual(expected_result['L'], query_result['var_11'], places=4)
        self.assertAlmostEqual(expected_result['M'], query_result['var_12'], places=4)
        self.assertAlmostEqual(expected_result['N'], query_result['var_13'], places=4)
        self.assertAlmostEqual(expected_result['O'], query_result['var_14'], places=4)
        self.assertAlmostEqual(expected_result['P'], query_result['var_15'], places=4)

        # The final Integrality gap after solving for the present case
        int_gap = self.mplp.get_integrality_gap()
        # Since the ties are broken arbitrary, we have 2 possible solutions howsoever trivial in difference
        self.assertIn(round(int_gap, 2), (7.98, 8.07))
