import numpy as np
import pytest

from pgmpy.inference.mplp import Mplp
from pgmpy.readwrite import UAIReader


@pytest.fixture
def mplp_instance():
    reader_file = UAIReader("pgmpy/tests/test_readwrite/testdata/grid4x4_with_triplets.uai")
    markov_model = reader_file.get_model()

    for factor in markov_model.factors:
        factor.values = np.log(factor.values)
    mplp = Mplp(markov_model)
    return mplp


class TestTightenTripletOff:
    # Query when tighten triplet is OFF
    def test_query_tighten_triplet_off(self, mplp_instance):
        mplp = mplp_instance
        query_result = mplp.map_query(tighten_triplet=False)

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
        assert query_result == expected_result

        # The final Integrality gap after solving for the present case
        int_gap = mplp.get_integrality_gap()
        assert round(int_gap, 1) == round(64.59, 1)


class TestTightenTripletOn:
    # Query when tighten triplet is ON
    def test_query_tighten_triplet_on(self, mplp_instance):
        mplp = mplp_instance
        query_result = mplp.map_query(tighten_triplet=True)
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

        assert query_result == expected_result

        # The final Integrality gap after solving for the present case
        int_gap = mplp.get_integrality_gap()
        # Since the ties are broken arbitrary, we have 2 possible solutions howsoever trivial in difference
        assert round(int_gap, 2) in (7.98, 8.07)
