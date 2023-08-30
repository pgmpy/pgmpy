import unittest

import numpy as np

from pgmpy import config
from pgmpy.readwrite import NETReader, NETWriter
from pgmpy.utils import compat_fns, get_example_model


class TestNETWriter(unittest.TestCase):
    def setUp(self):
        asia = get_example_model("asia")
        self.writer = NETWriter(asia)

    def test_get_variables(self):
        self.assertListEqual(
            self.writer.get_variables(),
            ["asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"],
        )

    def test_get_states(self):
        self.assertDictEqual(
            self.writer.get_states(),
            {
                "asia": ["yes", "no"],
                "bronc": ["yes", "no"],
                "dysp": ["yes", "no"],
                "either": ["yes", "no"],
                "lung": ["yes", "no"],
                "smoke": ["yes", "no"],
                "tub": ["yes", "no"],
                "xray": ["yes", "no"],
            },
        )

    def test_get_parents(self):
        self.assertDictEqual(
            self.writer.get_parents(),
            {
                "asia": [],
                "bronc": ["smoke"],
                "dysp": ["bronc", "either"],
                "either": ["lung", "tub"],
                "lung": ["smoke"],
                "smoke": [],
                "tub": ["asia"],
                "xray": ["either"],
            },
        )

    def test_get_cpds(self):
        cpds = self.writer.get_cpds()
        # np.testing.assert_array_equal returns None if equal
        self.assertIsNone(
            np.testing.assert_array_equal(cpds["asia"], np.array([0.01, 0.99]))
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                cpds["bronc"], np.array([[0.6, 0.3], [0.4, 0.7]])
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                cpds["dysp"],
                np.array([[[0.9, 0.8], [0.7, 0.1]], [[0.1, 0.2], [0.3, 0.9]]]),
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                cpds["either"],
                np.array([[[1.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]),
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                cpds["lung"], np.array([[0.1, 0.01], [0.9, 0.99]])
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(cpds["smoke"], np.array([0.5, 0.5]))
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                cpds["tub"], np.array([[0.05, 0.01], [0.95, 0.99]])
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                cpds["xray"], np.array([[0.98, 0.05], [0.02, 0.95]])
            )
        )

    def test_net_cpd(self):
        self.assertEqual(self.writer.net_cpd("asia"), "(0.01 0.99)")
        self.assertEqual(self.writer.net_cpd("bronc"), "((0.6 0.4)\n (0.3 0.7))")
        self.assertEqual(
            self.writer.net_cpd("dysp"),
            "(((0.9 0.1)\n  (0.8 0.2))\n\n ((0.7 0.3)\n  (0.1 0.9)))",
        )
        self.assertEqual(
            self.writer.net_cpd("either"),
            "(((1.0 0.0)\n  (1.0 0.0))\n\n ((1.0 0.0)\n  (0.0 1.0)))",
        )
        self.assertEqual(self.writer.net_cpd("lung"), "((0.1  0.9 )\n (0.01 0.99))")
        self.assertEqual(self.writer.net_cpd("smoke"), "(0.5 0.5)")
        self.assertEqual(self.writer.net_cpd("tub"), "((0.05 0.95)\n (0.01 0.99))")
        self.assertEqual(self.writer.net_cpd("xray"), "((0.98 0.02)\n (0.05 0.95))")

    def test_str(self):
        net = """net {
}
node asia{
    states = ("yes"  "no");
    weight = None;
}
node bronc{
    states = ("yes"  "no");
    weight = None;
}
node dysp{
    states = ("yes"  "no");
    weight = None;
}
node either{
    states = ("yes"  "no");
    weight = None;
}
node lung{
    states = ("yes"  "no");
    weight = None;
}
node smoke{
    states = ("yes"  "no");
    weight = None;
}
node tub{
    states = ("yes"  "no");
    weight = None;
}
node xray{
    states = ("yes"  "no");
    weight = None;
}
potential (asia |){
 data = (0.01 0.99);
}
potential (bronc | smoke){
 data = ((0.6 0.4)
 (0.3 0.7));
}
potential (dysp | bronc either){
 data = (((0.9 0.1)
  (0.8 0.2))

 ((0.7 0.3)
  (0.1 0.9)));
}
potential (either | lung tub){
 data = (((1.0 0.0)
  (1.0 0.0))

 ((1.0 0.0)
  (0.0 1.0)));
}
potential (lung | smoke){
 data = ((0.1  0.9 )
 (0.01 0.99));
}
potential (smoke |){
 data = (0.5 0.5);
}
potential (tub | asia){
 data = ((0.05 0.95)
 (0.01 0.99));
}
potential (xray | either){
 data = ((0.98 0.02)
 (0.05 0.95));
}
"""
        self.assertEqual(str(self.writer), net)


class TestNETReader(unittest.TestCase):
    def setUp(self):
        net = """
        /// Bayesian Network in the Hugin (.net) Format
        /// Produced by Genie Software
        /// Output Created Oct 26 15:50:51 2022
            net
            {
                node_size = (76 36);
            }

            node VisitToAsia
            {
                label = "Visit To Asia?";
                position = (94 246);
                states = ("NoVisit"  "Visit");
            }

            node Tuberculosis
            {
                label = "Tuberculosis?";
                position = (94 155);
                states = ("Absent"  "Present");
            }

            node Smoking
            {
                label = "Smoking?";
                position = (310 246);
                states = ("NonSmoker"  "Smoker");
            }

            node LungCancer
            {
                label = "Lung Cancer?";
                position = (241 155);
                states = ("Absent"  "Present");
            }

            node TbOrCa
            {
                label = "Tuberculosis or Lung Cancer?";
                position = (181 67);
                states = ("Nothing"  "CancerORTuberculosis");
            }

            node XRay
            {
                label = "X-Ray Result";
                position = (94 -18);
                states = ("Normal"  "Abnormal");
            }

            node Bronchitis
            {
                label = "Bronchitis?";
                position = (393 155);
                states = ("Absent"  "Present");
            }

            node Dyspnea
            {
                label = "Dyspnea?";
                position = (310 -18);
                states = ("Absent"  "Present");
            }

            potential (VisitToAsia |)
            {
                data = (0.99000000 0.01000000);
            }

            potential (Tuberculosis | VisitToAsia)
            {
                data = ((0.99000000 0.01000000)
                    (0.95000000 0.05000000));
            }

            potential (Smoking |)
            {
                data = (0.50000000 0.50000000);
            }

            potential (LungCancer | Smoking)
            {
                data = ((0.99000000 0.01000000)
                    (0.90000000 0.10000000));
            }

            potential (TbOrCa | Tuberculosis LungCancer)
            {
                data = (((1.00000000 0.00000000)
                    (0.00000000 1.00000000))
                    ((0.00000000 1.00000000)
                    (0.00000000 1.00000000)));
            }

            potential (XRay | TbOrCa)
            {
                data = ((0.95000000 0.05000000)
                    (0.02000000 0.98000000));
            }

            potential (Bronchitis | Smoking)
            {
                data = ((0.70000000 0.30000000)
                    (0.40000000 0.60000000));
            }

            potential (Dyspnea | TbOrCa Bronchitis)
            {
                data = (((0.90000000 0.10000000)
                    (0.20000000 0.80000000))
                    ((0.30000000 0.70000000)
                    (0.10000000 0.90000000)));
            }
            """
        self.reader = NETReader(string=net)

    def test_get_variables(self):
        var_expected = [
            "VisitToAsia",
            "Tuberculosis",
            "Smoking",
            "LungCancer",
            "TbOrCa",
            "XRay",
            "Bronchitis",
            "Dyspnea",
        ]
        self.assertListEqual(self.reader.get_variables(), var_expected)

    def test_get_states(self):
        states_expected = {
            "VisitToAsia": ["NoVisit", "Visit"],
            "Tuberculosis": ["Absent", "Present"],
            "Smoking": ["NonSmoker", "Smoker"],
            "LungCancer": ["Absent", "Present"],
            "TbOrCa": ["Nothing", "CancerORTuberculosis"],
            "XRay": ["Normal", "Abnormal"],
            "Bronchitis": ["Absent", "Present"],
            "Dyspnea": ["Absent", "Present"],
        }
        states = self.reader.get_states()
        for variable in states_expected:
            self.assertListEqual(states_expected[variable], states[variable])

    def test_get_parents(self):
        parents_expected = {
            "VisitToAsia": [],
            "Tuberculosis": ["VisitToAsia"],
            "Smoking": [],
            "LungCancer": ["Smoking"],
            "TbOrCa": ["Tuberculosis", "LungCancer"],
            "XRay": ["TbOrCa"],
            "Bronchitis": ["Smoking"],
            "Dyspnea": ["TbOrCa", "Bronchitis"],
        }
        parents = self.reader.get_parents()
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable], parents[variable])

    def test_get_values(self):
        values_expected = {
            "VisitToAsia": np.array([[0.99], [0.01]]),
            "Tuberculosis": np.array([[0.99, 0.95], [0.01, 0.05]]),
            "Smoking": np.array([[0.5], [0.5]]),
            "LungCancer": np.array([[0.99, 0.9], [0.01, 0.1]]),
            "TbOrCa": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]]),
            "XRay": np.array([[0.95, 0.02], [0.05, 0.98]]),
            "Bronchitis": np.array([[0.7, 0.4], [0.3, 0.6]]),
            "Dyspnea": np.array([[0.9, 0.2, 0.3, 0.1], [0.1, 0.8, 0.7, 0.9]]),
        }
        values = self.reader.get_values()
        for variable in values_expected:
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    values_expected[variable], values[variable]
                )
            )

    def test_get_edges(self):
        edges_expected = [
            ["VisitToAsia", "Tuberculosis"],
            ["Smoking", "LungCancer"],
            ["Tuberculosis", "TbOrCa"],
            ["LungCancer", "TbOrCa"],
            ["TbOrCa", "XRay"],
            ["Smoking", "Bronchitis"],
            ["TbOrCa", "Dyspnea"],
            ["Bronchitis", "Dyspnea"],
        ]
        edges = self.reader.get_edges()
        for index, edge in enumerate(edges_expected):
            self.assertListEqual(edge, edges[index])

    def test_get_properties(self):
        pass

    def test_get_network_name(self):
        pass


class TestNETWriterTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        asia = get_example_model("asia")
        self.writer = NETWriter(asia)

    def test_get_variables(self):
        self.assertListEqual(
            self.writer.get_variables(),
            ["asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"],
        )

    def test_get_states(self):
        self.assertDictEqual(
            self.writer.get_states(),
            {
                "asia": ["yes", "no"],
                "bronc": ["yes", "no"],
                "dysp": ["yes", "no"],
                "either": ["yes", "no"],
                "lung": ["yes", "no"],
                "smoke": ["yes", "no"],
                "tub": ["yes", "no"],
                "xray": ["yes", "no"],
            },
        )

    def test_get_parents(self):
        self.assertDictEqual(
            self.writer.get_parents(),
            {
                "asia": [],
                "bronc": ["smoke"],
                "dysp": ["bronc", "either"],
                "either": ["lung", "tub"],
                "lung": ["smoke"],
                "smoke": [],
                "tub": ["asia"],
                "xray": ["either"],
            },
        )

    def test_get_cpds(self):
        cpds = self.writer.get_cpds()
        np.testing.assert_array_equal(
            compat_fns.to_numpy(cpds["asia"], decimals=2), np.array([0.01, 0.99])
        )
        np.testing.assert_array_equal(
            compat_fns.to_numpy(cpds["bronc"], decimals=2),
            np.array([[0.6, 0.3], [0.4, 0.7]]),
        )
        np.testing.assert_array_equal(
            compat_fns.to_numpy(cpds["dysp"], decimals=2),
            np.array([[[0.9, 0.8], [0.7, 0.1]], [[0.1, 0.2], [0.3, 0.9]]]),
        )
        np.testing.assert_array_equal(
            compat_fns.to_numpy(cpds["either"], decimals=2),
            np.array([[[1.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]),
        )
        np.testing.assert_array_equal(
            compat_fns.to_numpy(cpds["lung"], decimals=2),
            np.array([[0.1, 0.01], [0.9, 0.99]]),
        )
        np.testing.assert_array_equal(
            compat_fns.to_numpy(cpds["smoke"], decimals=2), np.array([0.5, 0.5])
        )
        np.testing.assert_array_equal(
            compat_fns.to_numpy(cpds["tub"], decimals=2),
            np.array([[0.05, 0.01], [0.95, 0.99]]),
        )
        np.testing.assert_array_equal(
            compat_fns.to_numpy(cpds["xray"], decimals=2),
            np.array([[0.98, 0.05], [0.02, 0.95]]),
        )

    def test_net_cpd(self):
        self.assertEqual(self.writer.net_cpd("asia"), "(0.01 0.99)")
        self.assertEqual(self.writer.net_cpd("bronc"), "((0.6 0.4)\n (0.3 0.7))")
        self.assertEqual(
            self.writer.net_cpd("dysp"),
            "(((0.9 0.1)\n  (0.8 0.2))\n\n ((0.7 0.3)\n  (0.1 0.9)))",
        )
        self.assertEqual(
            self.writer.net_cpd("either"),
            "(((1.0 0.0)\n  (1.0 0.0))\n\n ((1.0 0.0)\n  (0.0 1.0)))",
        )
        self.assertEqual(self.writer.net_cpd("lung"), "((0.1  0.9 )\n (0.01 0.99))")
        self.assertEqual(self.writer.net_cpd("smoke"), "(0.5 0.5)")
        self.assertEqual(self.writer.net_cpd("tub"), "((0.05 0.95)\n (0.01 0.99))")
        self.assertEqual(self.writer.net_cpd("xray"), "((0.98 0.02)\n (0.05 0.95))")

    def test_str(self):
        net = """net {
}
node asia{
    states = ("yes"  "no");
    weight = None;
}
node bronc{
    states = ("yes"  "no");
    weight = None;
}
node dysp{
    states = ("yes"  "no");
    weight = None;
}
node either{
    states = ("yes"  "no");
    weight = None;
}
node lung{
    states = ("yes"  "no");
    weight = None;
}
node smoke{
    states = ("yes"  "no");
    weight = None;
}
node tub{
    states = ("yes"  "no");
    weight = None;
}
node xray{
    states = ("yes"  "no");
    weight = None;
}
potential (asia |){
 data = (0.01 0.99);
}
potential (bronc | smoke){
 data = ((0.6 0.4)
 (0.3 0.7));
}
potential (dysp | bronc either){
 data = (((0.9 0.1)
  (0.8 0.2))

 ((0.7 0.3)
  (0.1 0.9)));
}
potential (either | lung tub){
 data = (((1.0 0.0)
  (1.0 0.0))

 ((1.0 0.0)
  (0.0 1.0)));
}
potential (lung | smoke){
 data = ((0.1  0.9 )
 (0.01 0.99));
}
potential (smoke |){
 data = (0.5 0.5);
}
potential (tub | asia){
 data = ((0.05 0.95)
 (0.01 0.99));
}
potential (xray | either){
 data = ((0.98 0.02)
 (0.05 0.95));
}
"""
        self.assertEqual(str(self.writer), net)

    def tearDown(self):
        config.set_backend("numpy")


class TestNETReader(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        net = """
        /// Bayesian Network in the Hugin (.net) Format
        /// Produced by Genie Software
        /// Output Created Oct 26 15:50:51 2022
            net
            {
                node_size = (76 36);
            }

            node VisitToAsia
            {
                label = "Visit To Asia?";
                position = (94 246);
                states = ("NoVisit"  "Visit");
            }

            node Tuberculosis
            {
                label = "Tuberculosis?";
                position = (94 155);
                states = ("Absent"  "Present");
            }

            node Smoking
            {
                label = "Smoking?";
                position = (310 246);
                states = ("NonSmoker"  "Smoker");
            }

            node LungCancer
            {
                label = "Lung Cancer?";
                position = (241 155);
                states = ("Absent"  "Present");
            }

            node TbOrCa
            {
                label = "Tuberculosis or Lung Cancer?";
                position = (181 67);
                states = ("Nothing"  "CancerORTuberculosis");
            }

            node XRay
            {
                label = "X-Ray Result";
                position = (94 -18);
                states = ("Normal"  "Abnormal");
            }

            node Bronchitis
            {
                label = "Bronchitis?";
                position = (393 155);
                states = ("Absent"  "Present");
            }

            node Dyspnea
            {
                label = "Dyspnea?";
                position = (310 -18);
                states = ("Absent"  "Present");
            }

            potential (VisitToAsia |)
            {
                data = (0.99000000 0.01000000);
            }

            potential (Tuberculosis | VisitToAsia)
            {
                data = ((0.99000000 0.01000000)
                    (0.95000000 0.05000000));
            }

            potential (Smoking |)
            {
                data = (0.50000000 0.50000000);
            }

            potential (LungCancer | Smoking)
            {
                data = ((0.99000000 0.01000000)
                    (0.90000000 0.10000000));
            }

            potential (TbOrCa | Tuberculosis LungCancer)
            {
                data = (((1.00000000 0.00000000)
                    (0.00000000 1.00000000))
                    ((0.00000000 1.00000000)
                    (0.00000000 1.00000000)));
            }

            potential (XRay | TbOrCa)
            {
                data = ((0.95000000 0.05000000)
                    (0.02000000 0.98000000));
            }

            potential (Bronchitis | Smoking)
            {
                data = ((0.70000000 0.30000000)
                    (0.40000000 0.60000000));
            }

            potential (Dyspnea | TbOrCa Bronchitis)
            {
                data = (((0.90000000 0.10000000)
                    (0.20000000 0.80000000))
                    ((0.30000000 0.70000000)
                    (0.10000000 0.90000000)));
            }
            """
        self.reader = NETReader(string=net)

    def test_get_variables(self):
        var_expected = [
            "VisitToAsia",
            "Tuberculosis",
            "Smoking",
            "LungCancer",
            "TbOrCa",
            "XRay",
            "Bronchitis",
            "Dyspnea",
        ]
        self.assertListEqual(self.reader.get_variables(), var_expected)

    def test_get_states(self):
        states_expected = {
            "VisitToAsia": ["NoVisit", "Visit"],
            "Tuberculosis": ["Absent", "Present"],
            "Smoking": ["NonSmoker", "Smoker"],
            "LungCancer": ["Absent", "Present"],
            "TbOrCa": ["Nothing", "CancerORTuberculosis"],
            "XRay": ["Normal", "Abnormal"],
            "Bronchitis": ["Absent", "Present"],
            "Dyspnea": ["Absent", "Present"],
        }
        states = self.reader.get_states()
        for variable in states_expected:
            self.assertListEqual(states_expected[variable], states[variable])

    def test_get_parents(self):
        parents_expected = {
            "VisitToAsia": [],
            "Tuberculosis": ["VisitToAsia"],
            "Smoking": [],
            "LungCancer": ["Smoking"],
            "TbOrCa": ["Tuberculosis", "LungCancer"],
            "XRay": ["TbOrCa"],
            "Bronchitis": ["Smoking"],
            "Dyspnea": ["TbOrCa", "Bronchitis"],
        }
        parents = self.reader.get_parents()
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable], parents[variable])

    def test_get_values(self):
        values_expected = {
            "VisitToAsia": np.array([[0.99], [0.01]]),
            "Tuberculosis": np.array([[0.99, 0.95], [0.01, 0.05]]),
            "Smoking": np.array([[0.5], [0.5]]),
            "LungCancer": np.array([[0.99, 0.9], [0.01, 0.1]]),
            "TbOrCa": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]]),
            "XRay": np.array([[0.95, 0.02], [0.05, 0.98]]),
            "Bronchitis": np.array([[0.7, 0.4], [0.3, 0.6]]),
            "Dyspnea": np.array([[0.9, 0.2, 0.3, 0.1], [0.1, 0.8, 0.7, 0.9]]),
        }
        values = self.reader.get_values()
        for variable in values_expected:
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    values_expected[variable], values[variable]
                )
            )

    def test_get_edges(self):
        edges_expected = [
            ["VisitToAsia", "Tuberculosis"],
            ["Smoking", "LungCancer"],
            ["Tuberculosis", "TbOrCa"],
            ["LungCancer", "TbOrCa"],
            ["TbOrCa", "XRay"],
            ["Smoking", "Bronchitis"],
            ["TbOrCa", "Dyspnea"],
            ["Bronchitis", "Dyspnea"],
        ]
        edges = self.reader.get_edges()
        for index, edge in enumerate(edges_expected):
            self.assertListEqual(edge, edges[index])

    def test_get_properties(self):
        pass

    def test_get_network_name(self):
        pass

    def tearDown(self):
        config.set_backend("numpy")
