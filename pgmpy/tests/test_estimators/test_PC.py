import unittest

import pandas as pd
import numpy as np

from pgmpy.estimators import PC


class TestPCFakeCITest(unittest.TestCase):
    def setUp(self):
        self.fake_data = pd.DataFrame(np.random.random((1000, 4)), columns=['A', 'B', 'C', 'D'])
        self.estimator = PC(self.fake_data)

    def fake_ci_test(self, X, Y, Z=[]):
        """
        A mock CI testing function which gives False for every condition
        except for the following:
            1. B _|_ C
            2. B _|_ D
            3. C _|_ D
            4. A _|_ B | C
            5. A _|_ C | B
        """
        if X == "B":
            if Y == "C" or Y == "D":
                return True
            elif Y == "A" and Z == ["C"]:
                return True
        elif X == "C" and Y == "D" and Z == []:
            return True
        elif X == "D" and Y == "C" and Z == []:
            return True
        elif Y == "B":
            if X == "C" or X == "D":
                return True
            elif X == "A" and Z == ["C"]:
                return True
        elif X == 'A' and Y == 'C' and Z == ['B']:
            return True
        elif X == 'C' and Y == 'A' and Z == ['B']:
            return True
        return False

    def test_build_skeleton_orig(self):
        skel, sep_set = self.estimator.build_skeleton_orig(ci_test=self.fake_ci_test)
        expected_edges = {('A', 'C'), ('A', 'D')}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))

        skel, sep_set = self.estimator.build_skeleton_orig(ci_test=self.fake_ci_test, max_cond_vars=0)
        expected_edges = {('A', 'B'), ('A', 'C'), ('A', 'D')}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))

    def test_build_skeleton_stable(self):
        skel, sep_set = self.estimator.build_skeleton_stable(ci_test=self.fake_ci_test)
        expected_edges = {('A', 'D')}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))

        skel, sep_set = self.estimator.build_skeleton_orig(ci_test=self.fake_ci_test, max_cond_vars=0)
        expected_edges = {('A', 'B'), ('A', 'C'), ('A', 'D')}
        for u, v in skel.edges():
            self.assertTrue(((u, v) in expected_edges) or ((v, u) in expected_edges))


