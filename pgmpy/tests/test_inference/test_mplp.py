import unittest

import numpy as np

from pgmpy.inference.mplp import Mplp
from pgmpy.readwrite import UAIReader


class TestMplp(unittest.TestCase):
    def setUp(self):
        reader_file = UAIReader(
            "pgmpy/tests/test_readwrite/testdata/grid4x4_with_triplets.uai"
        )
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
            "var_1": 1,
            "var_0": 1,
            "var_2": 0,
            "var_3": 0,
            "var_4": 1,
            "var_5": 0,
            "var_6": 1,
            "var_7": 0,
            "var_8": 0,
            "var_9": 0,
            "var_10": 1,
            "var_11": 1,
            "var_12": 1,
            "var_13": 0,
            "var_14": 1,
            "var_15": 0,
        }
        self.assertEqual(query_result, expected_result)

        # The final Integrality gap after solving for the present case
        int_gap = self.mplp.get_integrality_gap()
        self.assertAlmostEqual(64.59, int_gap, places=1)


class TightenTripletOn(TestMplp):
    # Query when tighten triplet is ON
    def test_query_tighten_triplet_on(self):
        query_result = self.mplp.map_query(tighten_triplet=True)
        # Results from the Sontag code for a mplp run with tightening is:
        expected_result = {
            "var_0": 1,
            "var_1": 0,
            "var_2": 1,
            "var_3": 0,
            "var_4": 1,
            "var_5": 0,
            "var_6": 0,
            "var_7": 0,
            "var_8": 0,
            "var_9": 0,
            "var_10": 0,
            "var_11": 0,
            "var_12": 1,
            "var_13": 0,
            "var_14": 1,
            "var_15": 1,
        }

        self.assertEqual(query_result, expected_result)

        # The final Integrality gap after solving for the present case
        int_gap = self.mplp.get_integrality_gap()
        # Since the ties are broken arbitrary, we have 2 possible solutions howsoever trivial in difference
        self.assertIn(round(int_gap, 2), (7.98, 8.07))
