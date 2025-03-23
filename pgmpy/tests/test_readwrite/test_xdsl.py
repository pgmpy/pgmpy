import unittest
import xml.etree.ElementTree as etree

import numpy as np
import numpy.testing as np_test

from pgmpy import config
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite.XDSL import XDSLReader, XDSLWriter
from pgmpy.utils import get_example_model


class TestXDSLWriterMethodsString(unittest.TestCase):
    def setUp(self):
        self.asia_model_xdsl = "pgmpy\tests\test_readwrite\testdata\AsiaDiagnosis.xdsl"
        self.asia_model_bn = get_example_model(model="asia")

    def assert_models_equivalent(self, expected, got):
        self.assertSetEqual(set(expected.nodes()), set(got.nodes()))
        for node in expected.nodes():
            self.assertListEqual(
                sorted(expected.get_parents(node)), sorted(got.get_parents(node))
            )
            cpds_expected = expected.get_cpds(node=node)
            order = cpds_expected.variables[1:]
            cpds_got = got.get_cpds(node=node)

            self.assertEqual(
                cpds_expected.get_values().all(), cpds_got.get_values().all()
            )

    def test_write_xdsl(self):
        asia_xdsl = XDSLWriter(self.asia_model_bn).write_xdsl()
        asia_model_bn_test = XDSLReader(string=asia_xdsl).get_model()
        self.assert_models_equivalent(self.asia_model_bn, asia_model_bn_test)
