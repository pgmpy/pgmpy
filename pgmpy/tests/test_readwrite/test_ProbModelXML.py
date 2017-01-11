#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import warnings
import json

import numpy as np
import numpy.testing as np_test

from pgmpy.readwrite import ProbModelXMLReader, ProbModelXMLWriter, get_probmodel_data
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.extern.six.moves import range
from pgmpy.extern import six

try:
    from lxml import etree
except ImportError:
    try:
        import xml.etree.cElementTree as etree
    except ImportError:
        try:
            import xml.etree.ElementTree as etree
            print("running with ElementTree on Python 2.5+")
        except ImportError:
            warnings.warn("Failed to import ElementTree from any known place")


class TestProbModelXMLReaderString(unittest.TestCase):
    def setUp(self):
        string = """<ProbModelXML formatVersion="1.0">
      <ProbNet type="BayesianNetwork">
        <AdditionalConstraints>
            <Constraint name="MaxNumParents">
                <Argument name="numParents" value="5" />
            </Constraint>
        </AdditionalConstraints>
        <Comment>Student example model from Probabilistic Graphical Models: Principles and Techniques by Daphne Koller</Comment>
        <Language>English</Language>
        <AdditionalProperties>
            <Property name="elvira.title" value="X ray result"/>
        </AdditionalProperties>
        <DecisionCriteria>
            <Criterion name="cost">
                <AdditionalProperties/>
            </Criterion>
            <Criterion name="effectiveness">
                <AdditionalProperties/>
            </Criterion>
        </DecisionCriteria>
        <Variables>
            <Variable name="intelligence" type="FiniteState" role="Chance">
                <Comment />
                <Coordinates />
                <AdditionalProperties />
                <States>
                    <State name="smart"><AdditionalProperties /></State>
                    <State name="dumb"><AdditionalProperties /></State>
                </States>
            </Variable>
            <Variable name="difficulty" type="FiniteState" role="Chance">
                <Comment />
                <Coordinates />
                <AdditionalProperties />
                <States>
                    <State name="difficult"><AdditionalProperties /></State>
                    <State name="easy"><AdditionalProperties /></State>
                </States>
            </Variable>
        </Variables>
        <Links>
            <Link directed="1">
                <Variable name="difficulty" />
                <Variable name="grade" />
                <Comment>Directed Edge from difficulty to grade</Comment>
                <Label>diff_to_grad</Label>
                <AdditionalProperties />
            </Link>
            <Link directed="1">
                <Variable name="intelligence" />
                <Variable name="grade" />
                <Comment>Directed Edge from intelligence to grade</Comment>
                <Label>intel_to_grad</Label>
                <AdditionalProperties />
            </Link>
            <Link directed="1">
                <Variable name="intelligence" />
                <Variable name="SAT" />
                <Comment>Directed Edge from intelligence to SAT</Comment>
                <Label>intel_to_sat</Label>
                <AdditionalProperties />
            </Link>
            <Link directed="1">
                <Variable name="grade" />
                <Variable name="recommendation_letter" />
                <Comment>Directed Edge from grade to recommendation_letter</Comment>
                <Label>grad_to_reco</Label>
                <AdditionalProperties />
            </Link>
        </Links>
        <Potentials>
          <Potential type="Tree/ADD" role="Utility">
                <UtilityVariable name="U1" />
                <Variables>
                    <Variable name="D0"/>
                    <Variable name="D1"/>
                    <Variable name="C0"/>
                    <Variable name="C1"/>
                 </Variables>
                <TopVariable name="D0"/>
                <Branches>
                   <Branch>
                      <States><State name="no"/></States>
                      <Potential type="Tree/ADD">
                         <TopVariable name="C1"/>
                         <Branches>
                              <Branch>
                                  <Thresholds>
                                       <Threshold value="–Infinity"/>
                                       <Threshold value="0" belongsTo="Left"/>
                                  </Thresholds>
                                  <Potential type="MixtureOfExponentials">
                                      <Variables>
                                         <Variable name="C0"/>
                                         <Variable name="C1"/>
                                      </Variables>
                                      <Subpotentials>
                                         <Potential type="Exponential">
                                             <Potential type="Table">
                                                  <Values>3</Values>
                                              </Potential>
                                          </Potential>
                                          <Potential type="Exponential">
                                              <Potential type="Table">
                                                  <Values>–1</Values>
                                              </Potential>
                                              <NumericVariables>
                                                  <Variable name="C0"/>
                                                  <Variable name="C1"/>
                                              </NumericVariables>
                                              <Coefficients>4 –1</Coefficients>
                                           </Potential>
                                      </Subpotentials>
                                 </Potential>
                              </Branch>
                              <Branch>
                                    <Thresholds>
                                        <Threshold value="0" belongsTo="Left" />
                                        <Threshold value="+Infinity" />
                                     </Thresholds>
                                     <Potential type="MixtureOfExponentials">
                                        <Variables>
                                              <Variable name="C1"/>
                                              <Variable name="D1"/>
                                        </Variables>
                                        <Subpotentials>
                                              <Potential type="Exponential">
                                                 <Potential type="Table">
                                                      <Variables>
                                                          <Variable name="D1"/>
                                                      </Variables>
                                                      <Values>10  5</Values>
                                                 </Potential>
                                                 <NumericVariables>
                                                    <Variable name="C1"/>
                                                 </NumericVariables>
                                                 <Coefficients>0.25</Coefficients>
                                              </Potential>
                                         </Subpotentials>
                                     </Potential>
                              </Branch>
                         </Branches>
                      </Potential>
                   </Branch>
                  <Branch>
                        <States>
                            <State name="yes"/>
                         </States>
                         <Potential type="MixtureOfExponentials">
                                <Variables>
                                    <Variable name="C0"/>
                                </Variables>
                                <Subpotentials>
                                    <Potential type="Exponential">
                                        <Potential type="Table">
                                             <Values>0.3</Values>
                                        </Potential>
                                        <NumericVariables>
                                             <Variable name="C0"/>
                                        </NumericVariables>
                                        <Coefficients>1</Coefficients>
                                    </Potential>
                                   <Potential type="Exponential">
                                        <Potential type="Table">
                                             <Values>0.7</Values>
                                        </Potential>
                                   </Potential>
                                </Subpotentials>
                         </Potential>
                  </Branch>
                </Branches>
          </Potential>
      </Potentials>
    </ProbNet>
    <Policies />
    <InferenceOptions />
</ProbModelXML>
"""
        self.maxDiff = None
        self.reader_string = ProbModelXMLReader(string=string)
        self.reader_file = ProbModelXMLReader(path=six.StringIO(string))

    def test_comment(self):
        comment_expected = ("Student example model from Probabilistic Graphical Models: "
                            "Principles and Techniques by Daphne Koller")
        self.assertEqual(self.reader_string.probnet['Comment'], comment_expected)
        self.assertEqual(self.reader_file.probnet['Comment'], comment_expected)

    def test_variables(self):
        variables = {'difficulty':
                     {'Comment': None,
                      'Coordinates': {},
                      'role': 'Chance',
                      'type': 'FiniteState',
                      'States': {'difficult': {}, 'easy': {}}},
                     'intelligence':
                     {'Comment': None,
                      'Coordinates': {},
                      'role': 'Chance',
                      'type': 'FiniteState',
                      'States': {'smart': {}, 'dumb': {}}}}
        self.assertDictEqual(self.reader_string.probnet['Variables'], variables)
        self.assertDictEqual(self.reader_file.probnet['Variables'], variables)

    def test_edges(self):
        edge = {('grade', 'recommendation_letter'):
                {'directed': '1',
                 'Comment': 'Directed Edge from grade to recommendation_letter',
                 'Label': 'grad_to_reco'},
                ('intelligence', 'grade'):
                {'directed': '1',
                 'Comment': 'Directed Edge from intelligence to grade',
                 'Label': 'intel_to_grad'},
                ('difficulty', 'grade'):
                {'directed': '1',
                 'Comment': 'Directed Edge from difficulty to grade',
                 'Label': 'diff_to_grad'},
                ('intelligence', 'SAT'):
                {'directed': '1',
                 'Comment': 'Directed Edge from intelligence to SAT',
                 'Label': 'intel_to_sat'}}
        self.assertDictEqual(self.reader_string.probnet['edges'], edge)
        self.assertDictEqual(self.reader_file.probnet['edges'], edge)

    def test_additionalconstraints(self):
        additionalconstraints_expected = {'MaxNumParents':
                                          {'numParents': '5'}}
        self.assertDictEqual(self.reader_string.probnet['AdditionalConstraints'],
                             additionalconstraints_expected)
        self.assertDictEqual(self.reader_file.probnet['AdditionalConstraints'],
                             additionalconstraints_expected)

    def test_additionalproperties(self):
        additionalproperties_expected = {'elvira.title': 'X ray result'}
        self.assertDictEqual(self.reader_string.probnet['AdditionalProperties'],
                             additionalproperties_expected)
        self.assertDictEqual(self.reader_file.probnet['AdditionalProperties'],
                             additionalproperties_expected)

    def test_decisioncriteria(self):
        decisioncriteria_expected = {'effectiveness': {},
                                     'cost': {}}
        self.assertDictEqual(self.reader_string.probnet['DecisionCriteria'],
                             decisioncriteria_expected)
        self.assertDictEqual(self.reader_file.probnet['DecisionCriteria'],
                             decisioncriteria_expected)

    def test_potential(self):
        potential_expected = [{
            'role': 'Utility',
            'Variables': {'D0': ['D1', 'C0', 'C1']},
            'type': 'Tree/ADD',
            'UtilityVaribale': 'U1',
            'Branches': [{
                'Potential': {
                    'type': 'Tree/ADD',
                    'Branches': [{'Thresholds': [{'value': u'–Infinity'},
                                                 {'value': '0', 'belongsTo': 'Left'}],
                                  'Potential': {'Subpotentials': [{'Potential': {'type': 'Table',
                                                                                 'Values': '3'},
                                                                   'type': 'Exponential'},
                                                                  {'NumericVariables': ['C0', 'C1'],
                                                                   'Potential': {'type': 'Table',
                                                                                 'Values': u'–1'},
                                                                   'Coefficients': u'4 –1',
                                                                   'type': 'Exponential'}],
                                                'Variables': {'C0': ['C1']},
                                                'type': 'MixtureOfExponentials'}},
                                 {'Thresholds': [{'value': '0', 'belongsTo': 'Left'},
                                                 {'value': '+Infinity'}],
                                  'Potential': {'Subpotentials': [{'NumericVariables': ['C1'],
                                                                   'Potential': {'Variables': {'D1': []},
                                                                                 'type': 'Table',
                                                                                 'Values': '10  5'},
                                                                   'Coefficients': '0.25',
                                                                   'type': 'Exponential'}],
                                                'Variables': {'C1': ['D1']},
                                                'type': 'MixtureOfExponentials'}}],
                    'TopVariable': 'C1'},
                'States': [{'name': 'no'}]},
                {'Potential': {'Subpotentials': [{'NumericVariables': ['C0'],
                                                  'Potential': {'type': 'Table',
                                                                'Values': '0.3'},
                                                  'Coefficients': '1',
                                                  'type': 'Exponential'},
                                                 {'Potential': {'type': 'Table',
                                                                'Values': '0.7'},
                                                  'type': 'Exponential'}],
                               'Variables': {'C0': []},
                               'type': 'MixtureOfExponentials'},
                 'States': [{'name': 'yes'}]}],
            'TopVariable': 'D0'}]

        self.assertListEqual(self.reader_string.probnet['Potentials'],
                             potential_expected)
        self.assertListEqual(self.reader_file.probnet['Potentials'],
                             potential_expected)

    def test_get_model(self):
        string = """<ProbModelXML formatVersion="0.2.0">
  <ProbNet type="BayesianNetwork">
    <Comment>Student example model from Probabilistic Graphical Models: Principles and Techniques by Daphne Koller</Comment>
    <Variables>
      <Variable name="X-ray" type="finiteStates" role="chance" isInput="false">
        <Comment><![CDATA[Indica si el test de rayos X ha sido positivo]]></Comment>
        <Coordinates x="252" y="322" />
        <AdditionalProperties>
          <Property name="Relevance" value="7.0" />
          <Property name="Title" value="X" />
        </AdditionalProperties>
        <Precision>0.01</Precision>
        <States>
          <State name="no" />
          <State name="yes" />
        </States>
      </Variable>
      <Variable name="Bronchitis" type="finiteStates" role="chance" isInput="false">
        <Coordinates x="698" y="181" />
        <AdditionalProperties>
          <Property name="Relevance" value="7.0" />
          <Property name="Title" value="B" />
        </AdditionalProperties>
        <Precision>0.01</Precision>
        <States>
          <State name="no" />
          <State name="yes" />
        </States>
      </Variable>
      <Variable name="Dyspnea" type="finiteStates" role="chance" isInput="false">
        <Coordinates x="533" y="321" />
        <AdditionalProperties>
          <Property name="Relevance" value="7.0" />
          <Property name="Title" value="D" />
        </AdditionalProperties>
        <Precision>0.01</Precision>
        <States>
          <State name="no" />
          <State name="yes" />
        </States>
      </Variable>
      <Variable name="VisitToAsia" type="finiteStates" role="chance" isInput="false">
        <Coordinates x="290" y="58" />
        <AdditionalProperties>
          <Property name="Relevance" value="7.0" />
          <Property name="Title" value="A" />
        </AdditionalProperties>
        <Precision>0.01</Precision>
        <States>
          <State name="no" />
          <State name="yes" />
        </States>
      </Variable>
      <Variable name="Smoker" type="finiteStates" role="chance" isInput="false">
        <Coordinates x="568" y="52" />
        <AdditionalProperties>
          <Property name="Relevance" value="7.0" />
          <Property name="Title" value="S" />
        </AdditionalProperties>
        <Precision>0.01</Precision>
        <States>
          <State name="no" />
          <State name="yes" />
        </States>
      </Variable>
      <Variable name="LungCancer" type="finiteStates" role="chance" isInput="false">
        <Coordinates x="421" y="152" />
        <AdditionalProperties>
          <Property name="Relevance" value="7.0" />
          <Property name="Title" value="L" />
        </AdditionalProperties>
        <Precision>0.01</Precision>
        <States>
          <State name="no" />
          <State name="yes" />
        </States>
      </Variable>
      <Variable name="Tuberculosis" type="finiteStates" role="chance" isInput="false">
        <Coordinates x="201" y="150" />
        <AdditionalProperties>
          <Property name="Relevance" value="7.0" />
          <Property name="Title" value="T" />
        </AdditionalProperties>
        <Precision>0.01</Precision>
        <States>
          <State name="no" />
          <State name="yes" />
        </States>
      </Variable>
      <Variable name="TuberculosisOrCancer" type="finiteStates" role="chance" isInput="false">
        <Coordinates x="336" y="238" />
        <AdditionalProperties>
          <Property name="Relevance" value="7.0" />
          <Property name="Title" value="E" />
        </AdditionalProperties>
        <Precision>0.01</Precision>
        <States>
          <State name="no" />
          <State name="yes" />
        </States>
      </Variable>
    </Variables>
    <Links>
      <Link directed="true">
        <Variable name="Bronchitis" />
        <Variable name="Dyspnea" />
      </Link>
      <Link directed="true">
        <Variable name="VisitToAsia" />
        <Variable name="Tuberculosis" />
      </Link>
      <Link directed="true">
        <Variable name="Smoker" />
        <Variable name="Bronchitis" />
      </Link>
      <Link directed="true">
        <Variable name="Smoker" />
        <Variable name="LungCancer" />
      </Link>
      <Link directed="true">
        <Variable name="LungCancer" />
        <Variable name="TuberculosisOrCancer" />
      </Link>
      <Link directed="true">
        <Variable name="Tuberculosis" />
        <Variable name="TuberculosisOrCancer" />
      </Link>
      <Link directed="true">
        <Variable name="TuberculosisOrCancer" />
        <Variable name="Dyspnea" />
      </Link>
      <Link directed="true">
        <Variable name="TuberculosisOrCancer" />
        <Variable name="X-ray" />
      </Link>
    </Links>
    <Potentials>
      <Potential type="Table" role="conditionalProbability">
        <Variables>
          <Variable name="X-ray" />
          <Variable name="TuberculosisOrCancer" />
        </Variables>
        <Values>0.95 0.05 0.02 0.98</Values>
      </Potential>
      <Potential type="Table" role="conditionalProbability">
        <Variables>
          <Variable name="Bronchitis" />
          <Variable name="Smoker" />
        </Variables>
        <Values>0.7 0.3 0.4 0.6</Values>
      </Potential>
      <Potential type="Table" role="conditionalProbability">
        <Variables>
          <Variable name="Dyspnea" />
          <Variable name="TuberculosisOrCancer" />
          <Variable name="Bronchitis" />
        </Variables>
        <Values>0.9 0.1 0.3 0.7 0.2 0.8 0.1 0.9</Values>
      </Potential>
      <Potential type="Table" role="conditionalProbability">
        <Variables>
          <Variable name="VisitToAsia" />
        </Variables>
        <Values>0.99 0.01</Values>
      </Potential>
      <Potential type="Table" role="conditionalProbability">
        <Variables>
          <Variable name="Smoker" />
        </Variables>
        <Values>0.5 0.5</Values>
      </Potential>
      <Potential type="Table" role="conditionalProbability">
        <Variables>
          <Variable name="LungCancer" />
          <Variable name="Smoker" />
        </Variables>
        <Values>0.99 0.01 0.9 0.1</Values>
      </Potential>
      <Potential type="Table" role="conditionalProbability">
        <Variables>
          <Variable name="Tuberculosis" />
          <Variable name="VisitToAsia" />
        </Variables>
        <Values>0.99 0.01 0.95 0.05</Values>
      </Potential>
      <Potential type="Table" role="conditionalProbability">
        <Variables>
          <Variable name="TuberculosisOrCancer" />
          <Variable name="LungCancer" />
          <Variable name="Tuberculosis" />
        </Variables>
        <Values>1.0 0.0 0.0 1.0 0.0 1.0 0.0 1.0</Values>
      </Potential>
    </Potentials>
    <AdditionalProperties>
      <Property name="VisualPrecision" value="0.0" />
      <Property name="KindOfGraph" value="directed" />
      <Property name="WhenChanged" value="19/08/99" />
      <Property name="Version" value="1.0" />
      <Property name="WhoChanged" value="Jose A. Gamez" />
    </AdditionalProperties>
  </ProbNet>
</ProbModelXML>"""
        self.maxDiff = None
        self.reader = ProbModelXMLReader(string=string)
        model = self.reader.get_model()
        edges_expected = [('VisitToAsia', 'Tuberculosis'),
                          ('LungCancer', 'TuberculosisOrCancer'),
                          ('Smoker', 'LungCancer'),
                          ('Smoker', 'Bronchitis'),
                          ('Tuberculosis', 'TuberculosisOrCancer'),
                          ('Bronchitis', 'Dyspnea'),
                          ('TuberculosisOrCancer', 'Dyspnea'),
                          ('TuberculosisOrCancer', 'X-ray')]
        node_expected = {'Smoker': {'States': {'no': {}, 'yes': {}},
                                    'role': 'chance',
                                    'type': 'finiteStates',
                                    'Coordinates': {'y': '52', 'x': '568'},
                                    'AdditionalProperties': {'Title': 'S', 'Relevance': '7.0'}},
                         'Bronchitis': {'States': {'no': {}, 'yes': {}},
                                        'role': 'chance',
                                        'type': 'finiteStates',
                                        'Coordinates': {'y': '181', 'x': '698'},
                                        'AdditionalProperties': {'Title': 'B', 'Relevance': '7.0'}},
                         'VisitToAsia': {'States': {'no': {}, 'yes': {}},
                                         'role': 'chance',
                                         'type': 'finiteStates',
                                         'Coordinates': {'y': '58', 'x': '290'},
                                         'AdditionalProperties': {'Title': 'A', 'Relevance': '7.0'}},
                         'Tuberculosis': {'States': {'no': {}, 'yes': {}},
                                          'role': 'chance',
                                          'type': 'finiteStates',
                                          'Coordinates': {'y': '150', 'x': '201'},
                                          'AdditionalProperties': {'Title': 'T', 'Relevance': '7.0'}},
                         'X-ray': {'States': {'no': {}, 'yes': {}},
                                   'role': 'chance',
                                   'AdditionalProperties': {'Title': 'X', 'Relevance': '7.0'},
                                   'Coordinates': {'y': '322', 'x': '252'},
                                   'Comment': 'Indica si el test de rayos X ha sido positivo',
                                   'type': 'finiteStates'},
                         'Dyspnea': {'States': {'no': {}, 'yes': {}},
                                     'role': 'chance',
                                     'type': 'finiteStates',
                                     'Coordinates': {'y': '321', 'x': '533'},
                                     'AdditionalProperties': {'Title': 'D', 'Relevance': '7.0'}},
                         'TuberculosisOrCancer': {'States': {'no': {}, 'yes': {}},
                                                  'role': 'chance',
                                                  'type': 'finiteStates',
                                                  'Coordinates': {'y': '238', 'x': '336'},
                                                  'AdditionalProperties': {'Title': 'E', 'Relevance': '7.0'}},
                         'LungCancer': {'States': {'no': {}, 'yes': {}},
                                        'role': 'chance',
                                        'type': 'finiteStates',
                                        'Coordinates': {'y': '152', 'x': '421'},
                                        'AdditionalProperties': {'Title': 'L', 'Relevance': '7.0'}}}
        edge_expected = {'LungCancer': {'TuberculosisOrCancer': {'weight': None,
                                                                 'directed': 'true'}},
                         'Smoker': {'LungCancer': {'weight': None,
                                                   'directed': 'true'},
                                    'Bronchitis': {'weight': None,
                                                   'directed': 'true'}},
                         'Dyspnea': {},
                         'X-ray': {},
                         'VisitToAsia': {'Tuberculosis': {'weight': None,
                                                          'directed': 'true'}},
                         'TuberculosisOrCancer': {'X-ray': {'weight': None,
                                                            'directed': 'true'},
                                                  'Dyspnea': {'weight': None,
                                                              'directed': 'true'}},
                         'Bronchitis': {'Dyspnea': {'weight': None,
                                                    'directed': 'true'}},
                         'Tuberculosis': {'TuberculosisOrCancer': {'weight': None,
                                                                   'directed': 'true'}}}

        cpds_expected = [np.array([[0.95, 0.05], [0.02, 0.98]]),
                         np.array([[0.7, 0.3], [0.4,  0.6]]),
                         np.array([[0.9, 0.1,  0.3,  0.7], [0.2,  0.8,  0.1,  0.9]]),
                         np.array([[0.99], [0.01]]),
                         np.array([[0.5], [0.5]]),
                         np.array([[0.99, 0.01], [0.9, 0.1]]),
                         np.array([[0.99, 0.01], [0.95, 0.05]]),
                         np.array([[1, 0, 0, 1], [0, 1, 0, 1]])]
        for cpd_index in range(0, len(cpds_expected)):
            np_test.assert_array_equal(model.get_cpds()[cpd_index].get_values(),
                                       cpds_expected[cpd_index])
        self.assertDictEqual(model.node, node_expected)
        self.assertDictEqual(model.edge, edge_expected)
        self.assertListEqual(sorted(model.edges()), sorted(edges_expected))


class TestProbModelXMLWriter(unittest.TestCase):
    def setUp(self):
        self.model_data = {'probnet':
                           {'type': 'BayesianNetwork',
                            'Language': 'English',
                            'AdditionalConstraints': {'MaxNumParents':
                                                      {'numParents': '5'}},
                            'AdditionalProperties': {'elvira.title': 'X ray result'},
                            'DecisionCriteria': {'effectiveness': {},
                                                 'cost': {}},
                            'Variables': {'difficulty':
                                          {'type': 'FiniteState',
                                           'role': 'Chance',
                                           'States': {'difficult': {}, 'easy': {}},
                                           'Comment': None,
                                           'Coordinates': {}},
                                          'intelligence':
                                          {'type': 'FiniteState',
                                           'role': 'Chance',
                                           'States': {'smart': {}, 'dumb': {}},
                                           'Comment': None,
                                           'Coordinates': {}}},
                            'Comment': 'Student example model from Probabilistic Graphical Models: '
                                       'Principles and Techniques by Daphne Koller',
                            'edges': {('difficulty', 'grade'):
                                      {'directed': '1',
                                       'Label': 'diff_to_grad',
                                       'Comment': 'Directed Edge from difficulty to grade'},
                                      ('intelligence', 'grade'):
                                      {'directed': '1',
                                       'Label': 'intel_to_grad',
                                       'Comment': 'Directed Edge from intelligence to grade'},
                                      ('intelligence', 'SAT'):
                                      {'directed': '1',
                                       'Label': 'intel_to_sat',
                                       'Comment': 'Directed Edge from intelligence to SAT'},
                                      ('grade', 'recommendation_letter'):
                                      {'directed': '1',
                                       'Label': 'grad_to_reco',
                                       'Comment': 'Directed Edge from grade to recommendation_letter'}},
                            'Potentials': [{'role': 'Utility',
                                            'Variables': {'D0': ['D1', 'C0', 'C1']},
                                            'type': 'Tree/ADD',
                                            'UtilityVaribale': 'U1',
                                            'Branches': [{
                                                'Potential': {
                                                    'type': 'Tree/ADD',
                                                    'Branches': [{
                                                        'Thresholds': [{'value': '-Infinity'},
                                                                       {'value': '0', 'belongsTo': 'Left'}],
                                                        'Potential': {'Subpotentials': [
                                                            {'Potential': {'type': 'Table',
                                                                           'Values': '3'},
                                                             'type': 'Exponential'},
                                                            {'NumericVariables': ['C0', 'C1'],
                                                             'Potential': {'type': 'Table',
                                                                           'Values': '-1'},
                                                             'Coefficients': '4 -1',
                                                             'type': 'Exponential'}],
                                                            'Variables': {'C0': ['C1']},
                                                            'type': 'MixtureOfExponentials'}},
                                                        {'Thresholds': [{'value': '0', 'belongsTo': 'Left'},
                                                                        {'value': '+Infinity'}],
                                                         'Potential': {'Subpotentials': [
                                                             {'NumericVariables': ['C1'],
                                                              'Potential': {'Variables': {'D1': []},
                                                                            'type': 'Table',
                                                                            'Values': '10  5'},
                                                              'Coefficients': '0.25',
                                                              'type': 'Exponential'}],
                                                             'Variables': {'C1': ['D1']},
                                                             'type': 'MixtureOfExponentials'}}],
                                                    'TopVariable': 'C1'},
                                                'States': [{'name': 'no'}]},
                                                {'Potential': {'Subpotentials': [
                                                    {'NumericVariables': ['C0'],
                                                     'Potential': {'type': 'Table',
                                                                   'Values': '0.3'},
                                                     'Coefficients': '1',
                                                     'type': 'Exponential'},
                                                    {'Potential': {'type': 'Table',
                                                                   'Values': '0.7'},
                                                     'type': 'Exponential'}],
                                                    'Variables': {'C0': []},
                                                    'type': 'MixtureOfExponentials'},
                                                    'States': [{'name': 'yes'}]}],
                                            'TopVariable': 'D0'}]}}

        self.maxDiff = None
        self.writer = ProbModelXMLWriter(model_data=self.model_data)

    def test_file(self):
        self.expected_xml = etree.XML("""<ProbModelXML formatVersion="1.0">
  <ProbNet type="BayesianNetwork">
    <Variables>
      <Variable name="difficulty" role="Chance" type="FiniteState">
        <Comment/>
        <Coordinates/>
        <AdditionalProperties/>
        <States>
          <State name="difficult">
            <AdditionalProperties/>
          </State>
          <State name="easy">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
      <Variable name="intelligence" role="Chance" type="FiniteState">
        <Comment/>
        <Coordinates/>
        <AdditionalProperties/>
        <States>
          <State name="dumb">
            <AdditionalProperties/>
          </State>
          <State name="smart">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
    </Variables>
    <Links>
      <Link directed="1" var1="difficulty" var2="grade">
        <Comment>Directed Edge from difficulty to grade</Comment>
        <Label>diff_to_grad</Label>
        <AdditionalProperties/>
      </Link>
      <Link directed="1" var1="grade" var2="recommendation_letter">
        <Comment>Directed Edge from grade to recommendation_letter</Comment>
        <Label>grad_to_reco</Label>
        <AdditionalProperties/>
      </Link>
      <Link directed="1" var1="intelligence" var2="SAT">
        <Comment>Directed Edge from intelligence to SAT</Comment>
        <Label>intel_to_sat</Label>
        <AdditionalProperties/>
      </Link>
      <Link directed="1" var1="intelligence" var2="grade">
        <Comment>Directed Edge from intelligence to grade</Comment>
        <Label>intel_to_grad</Label>
        <AdditionalProperties/>
      </Link>
    </Links>
    <Potentials>
      <Potential role="Utility" type="Tree/ADD">
        <Variables>
          <Variable name="D0"/>
          <Variable name="C0"/>
          <Variable name="C1"/>
          <Variable name="D1"/>
        </Variables>
        <TopVariable name="D0"/>
        <Branches>
          <Branch>
            <States>
              <State name="no"/>
            </States>
            <Potential type="Tree/ADD">
              <TopVariable name="C1"/>
              <Branches>
                <Branch>
                  <Potential type="MixtureOfExponentials">
                    <Variables>
                      <Variable name="C0"/>
                      <Variable name="C1"/>
                    </Variables>
                    <Subpotentials>
                      <Potential type="Exponential">
                        <Potential type="Table">
                          <Values>3</Values>
                        </Potential>
                      </Potential>
                      <Potential type="Exponential">
                        <Coefficients>4 -1</Coefficients>
                        <Potential type="Table">
                          <Values>-1</Values>
                        </Potential>
                        <NumericVariables>
                          <Variable name="C0"/>
                          <Variable name="C1"/>
                        </NumericVariables>
                      </Potential>
                    </Subpotentials>
                  </Potential>
                  <Thresholds>
                    <Threshold value="-Infinity"/>
                    <Threshold belongsTo="Left" value="0"/>
                  </Thresholds>
                </Branch>
                <Branch>
                  <Potential type="MixtureOfExponentials">
                    <Variables>
                      <Variable name="C1"/>
                      <Variable name="D1"/>
                    </Variables>
                    <Subpotentials>
                      <Potential type="Exponential">
                        <Coefficients>0.25</Coefficients>
                        <Potential type="Table">
                          <Variables>
                            <Variable name="D1"/>
                          </Variables>
                          <Values>10  5</Values>
                        </Potential>
                        <NumericVariables>
                          <Variable name="C1"/>
                        </NumericVariables>
                      </Potential>
                    </Subpotentials>
                  </Potential>
                  <Thresholds>
                    <Threshold belongsTo="Left" value="0"/>
                    <Threshold value="+Infinity"/>
                  </Thresholds>
                </Branch>
              </Branches>
            </Potential>
          </Branch>
          <Branch>
            <States>
              <State name="yes"/>
            </States>
            <Potential type="MixtureOfExponentials">
              <Variables>
                <Variable name="C0"/>
              </Variables>
              <Subpotentials>
                <Potential type="Exponential">
                  <Coefficients>1</Coefficients>
                  <Potential type="Table">
                    <Values>0.3</Values>
                  </Potential>
                  <NumericVariables>
                    <Variable name="C0"/>
                  </NumericVariables>
                </Potential>
                <Potential type="Exponential">
                  <Potential type="Table">
                    <Values>0.7</Values>
                  </Potential>
                </Potential>
              </Subpotentials>
            </Potential>
          </Branch>
        </Branches>
      </Potential>
    </Potentials>
    <AdditionalConstraints>
      <Constraint name="MaxNumParents">
        <Argument name="numParents" value="5"/>
      </Constraint>
    </AdditionalConstraints>
    <Language>English</Language>
    <Comment>Student example model from Probabilistic Graphical Models: Principles and Techniques by Daphne Koller</Comment>
  </ProbNet>
  <AdditionalProperties>
    <Property name="elvira.title" value="X ray result"/>
  </AdditionalProperties>
  <DecisionCriteria>
    <Criterion name="cost">
      <AdditionalProperties/>
    </Criterion>
    <Criterion name="effectiveness">
      <AdditionalProperties/>
    </Criterion>
  </DecisionCriteria>
</ProbModelXML>""")
        self.assertEqual(str(self.writer.__str__()[:-1]), str(etree.tostring(self.expected_xml)))

    def test_write_file(self):
        self.expected_xml = etree.XML("""<ProbModelXML formatVersion="1.0">
  <ProbNet type="BayesianNetwork">
    <Variables>
      <Variable name="difficulty" role="Chance" type="FiniteState">
        <Comment/>
        <Coordinates/>
        <AdditionalProperties/>
        <States>
          <State name="difficult">
            <AdditionalProperties/>
          </State>
          <State name="easy">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
      <Variable name="intelligence" role="Chance" type="FiniteState">
        <Comment/>
        <Coordinates/>
        <AdditionalProperties/>
        <States>
          <State name="dumb">
            <AdditionalProperties/>
          </State>
          <State name="smart">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
    </Variables>
    <Links>
      <Link directed="1" var1="difficulty" var2="grade">
        <Comment>Directed Edge from difficulty to grade</Comment>
        <Label>diff_to_grad</Label>
        <AdditionalProperties/>
      </Link>
      <Link directed="1" var1="grade" var2="recommendation_letter">
        <Comment>Directed Edge from grade to recommendation_letter</Comment>
        <Label>grad_to_reco</Label>
        <AdditionalProperties/>
      </Link>
      <Link directed="1" var1="intelligence" var2="SAT">
        <Comment>Directed Edge from intelligence to SAT</Comment>
        <Label>intel_to_sat</Label>
        <AdditionalProperties/>
      </Link>
      <Link directed="1" var1="intelligence" var2="grade">
        <Comment>Directed Edge from intelligence to grade</Comment>
        <Label>intel_to_grad</Label>
        <AdditionalProperties/>
      </Link>
    </Links>
    <Potentials>
      <Potential role="Utility" type="Tree/ADD">
        <Variables>
          <Variable name="D0"/>
          <Variable name="C0"/>
          <Variable name="C1"/>
          <Variable name="D1"/>
        </Variables>
        <TopVariable name="D0"/>
        <Branches>
          <Branch>
            <States>
              <State name="no"/>
            </States>
            <Potential type="Tree/ADD">
              <TopVariable name="C1"/>
              <Branches>
                <Branch>
                  <Potential type="MixtureOfExponentials">
                    <Variables>
                      <Variable name="C0"/>
                      <Variable name="C1"/>
                    </Variables>
                    <Subpotentials>
                      <Potential type="Exponential">
                        <Potential type="Table">
                          <Values>3</Values>
                        </Potential>
                      </Potential>
                      <Potential type="Exponential">
                        <Coefficients>4 -1</Coefficients>
                        <Potential type="Table">
                          <Values>-1</Values>
                        </Potential>
                        <NumericVariables>
                          <Variable name="C0"/>
                          <Variable name="C1"/>
                        </NumericVariables>
                      </Potential>
                    </Subpotentials>
                  </Potential>
                  <Thresholds>
                    <Threshold value="-Infinity"/>
                    <Threshold belongsTo="Left" value="0"/>
                  </Thresholds>
                </Branch>
                <Branch>
                  <Potential type="MixtureOfExponentials">
                    <Variables>
                      <Variable name="C1"/>
                      <Variable name="D1"/>
                    </Variables>
                    <Subpotentials>
                      <Potential type="Exponential">
                        <Coefficients>0.25</Coefficients>
                        <Potential type="Table">
                          <Variables>
                            <Variable name="D1"/>
                          </Variables>
                          <Values>10  5</Values>
                        </Potential>
                        <NumericVariables>
                          <Variable name="C1"/>
                        </NumericVariables>
                      </Potential>
                    </Subpotentials>
                  </Potential>
                  <Thresholds>
                    <Threshold belongsTo="Left" value="0"/>
                    <Threshold value="+Infinity"/>
                  </Thresholds>
                </Branch>
              </Branches>
            </Potential>
          </Branch>
          <Branch>
            <States>
              <State name="yes"/>
            </States>
            <Potential type="MixtureOfExponentials">
              <Variables>
                <Variable name="C0"/>
              </Variables>
              <Subpotentials>
                <Potential type="Exponential">
                  <Coefficients>1</Coefficients>
                  <Potential type="Table">
                    <Values>0.3</Values>
                  </Potential>
                  <NumericVariables>
                    <Variable name="C0"/>
                  </NumericVariables>
                </Potential>
                <Potential type="Exponential">
                  <Potential type="Table">
                    <Values>0.7</Values>
                  </Potential>
                </Potential>
              </Subpotentials>
            </Potential>
          </Branch>
        </Branches>
      </Potential>
    </Potentials>
    <AdditionalConstraints>
      <Constraint name="MaxNumParents">
        <Argument name="numParents" value="5"/>
      </Constraint>
    </AdditionalConstraints>
    <Language>English</Language>
    <Comment>Student example model from Probabilistic Graphical Models: Principles and Techniques by Daphne Koller</Comment>
  </ProbNet>
  <AdditionalProperties>
    <Property name="elvira.title" value="X ray result"/>
  </AdditionalProperties>
  <DecisionCriteria>
    <Criterion name="cost">
      <AdditionalProperties/>
    </Criterion>
    <Criterion name="effectiveness">
      <AdditionalProperties/>
    </Criterion>
  </DecisionCriteria>
</ProbModelXML>""")
        self.writer.write_file("test_xml.pgmx")
        with open("test_xml.pgmx", "r") as myfile:
            data = myfile.read()
        self.assertEqual(str(self.writer.__str__()[:-1]), str(etree.tostring(self.expected_xml)))
        self.assertEqual(str(data), str(etree.tostring(self.expected_xml).decode('utf-8')))


class TestProbModelXMLmethods(unittest.TestCase):
    def setUp(self):
        edges_list = [('VisitToAsia', 'Tuberculosis'),
                      ('LungCancer', 'TuberculosisOrCancer'),
                      ('Smoker', 'LungCancer'),
                      ('Smoker', 'Bronchitis'),
                      ('Tuberculosis', 'TuberculosisOrCancer'),
                      ('Bronchitis', 'Dyspnea'),
                      ('TuberculosisOrCancer', 'Dyspnea'),
                      ('TuberculosisOrCancer', 'X-ray')]
        nodes = {'Smoker': {'States': {'no': {}, 'yes': {}},
                            'role': 'chance',
                            'type': 'finiteStates',
                            'Coordinates': {'y': '52', 'x': '568'},
                            'AdditionalProperties': {'Title': 'S', 'Relevance': '7.0'}},
                 'Bronchitis': {'States': {'no': {}, 'yes': {}},
                                'role': 'chance',
                                'type': 'finiteStates',
                                'Coordinates': {'y': '181', 'x': '698'},
                                'AdditionalProperties': {'Title': 'B', 'Relevance': '7.0'}},
                 'VisitToAsia': {'States': {'no': {}, 'yes': {}},
                                 'role': 'chance',
                                 'type': 'finiteStates',
                                 'Coordinates': {'y': '58', 'x': '290'},
                                 'AdditionalProperties': {'Title': 'A', 'Relevance': '7.0'}},
                 'Tuberculosis': {'States': {'no': {}, 'yes': {}},
                                  'role': 'chance',
                                  'type': 'finiteStates',
                                  'Coordinates': {'y': '150', 'x': '201'},
                                  'AdditionalProperties': {'Title': 'T', 'Relevance': '7.0'}},
                 'X-ray': {'States': {'no': {}, 'yes': {}},
                           'role': 'chance',
                           'AdditionalProperties': {'Title': 'X', 'Relevance': '7.0'},
                           'Coordinates': {'y': '322', 'x': '252'},
                           'Comment': 'Indica si el test de rayos X ha sido positivo',
                           'type': 'finiteStates'},
                 'Dyspnea': {'States': {'no': {}, 'yes': {}},
                             'role': 'chance',
                             'type': 'finiteStates',
                             'Coordinates': {'y': '321', 'x': '533'},
                             'AdditionalProperties': {'Title': 'D', 'Relevance': '7.0'}},
                 'TuberculosisOrCancer': {'States': {'no': {}, 'yes': {}},
                                          'role': 'chance',
                                          'type': 'finiteStates',
                                          'Coordinates': {'y': '238', 'x': '336'},
                                          'AdditionalProperties': {'Title': 'E', 'Relevance': '7.0'}},
                 'LungCancer': {'States': {'no': {}, 'yes': {}},
                                'role': 'chance',
                                'type': 'finiteStates',
                                'Coordinates': {'y': '152', 'x': '421'},
                                'AdditionalProperties': {'Title': 'L', 'Relevance': '7.0'}}}
        edges = {'LungCancer': {'TuberculosisOrCancer': {'directed': 'true'}},
                 'Smoker': {'LungCancer': {'directed': 'true'},
                            'Bronchitis': {'directed': 'true'}},
                 'Dyspnea': {},
                 'X-ray': {},
                 'VisitToAsia': {'Tuberculosis': {'directed': 'true'}},
                 'TuberculosisOrCancer': {'X-ray': {'directed': 'true'},
                                          'Dyspnea': {'directed': 'true'}},
                 'Bronchitis': {'Dyspnea': {'directed': 'true'}},
                 'Tuberculosis': {'TuberculosisOrCancer': {'directed': 'true'}}}

        cpds = [{'Values': np.array([[0.95, 0.05], [0.02, 0.98]]),
                 'Variables': {'X-ray': ['TuberculosisOrCancer']}},
                {'Values': np.array([[0.7, 0.3], [0.4,  0.6]]),
                 'Variables': {'Bronchitis': ['Smoker']}},
                {'Values':  np.array([[0.9, 0.1,  0.3,  0.7], [0.2,  0.8,  0.1,  0.9]]),
                 'Variables': {'Dyspnea': ['TuberculosisOrCancer', 'Bronchitis']}},
                {'Values': np.array([[0.99], [0.01]]),
                 'Variables': {'VisitToAsia': []}},
                {'Values': np.array([[0.5], [0.5]]),
                 'Variables': {'Smoker': []}},
                {'Values': np.array([[0.99, 0.01], [0.9, 0.1]]),
                 'Variables': {'LungCancer': ['Smoker']}},
                {'Values': np.array([[0.99, 0.01], [0.95, 0.05]]),
                 'Variables': {'Tuberculosis': ['VisitToAsia']}},
                {'Values': np.array([[1, 0, 0, 1], [0, 1, 0, 1]]),
                 'Variables': {'TuberculosisOrCancer': ['LungCancer', 'Tuberculosis']}}]
        self.model = BayesianModel(edges_list)
        for node in nodes:
            self.model.node[node] = nodes[node]
        for edge in edges:
            self.model.edge[edge] = edges[edge]

        tabular_cpds = []
        for cpd in cpds:
            var = list(cpd['Variables'].keys())[0]
            evidence = cpd['Variables'][var]
            values = cpd['Values']
            states = len(nodes[var]['States'])
            evidence_card = [len(nodes[evidence_var]['States'])
                             for evidence_var in evidence]
            tabular_cpds.append(
                TabularCPD(var, states, values, evidence, evidence_card))
        self.maxDiff = None
        self.model.add_cpds(*tabular_cpds)

    def test_get_probmodel_data(self):
        model_data = get_probmodel_data(self.model)
        xmlfile = ProbModelXMLWriter(model_data)
        with open('pgmpy/tests/test_readwrite/testdata/test_probmodelxml_data.json') as data_file:
            model_data_expected = json.load(data_file)
        xmlfile_expected = etree.XML("""<ProbModelXML formatVersion="1.0">
  <ProbNet type="BayesianNetwork">
    <Variables>
      <Variable name="Bronchitis" role="chance" type="finiteStates">
        <Coordinates x="698" y="181"/>
        <Property name="Relevance" value="7.0"/>
        <Property name="Title" value="B"/>
        <States>
          <State name="no">
            <AdditionalProperties/>
          </State>
          <State name="yes">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
      <Variable name="Dyspnea" role="chance" type="finiteStates">
        <Coordinates x="533" y="321"/>
        <Property name="Relevance" value="7.0"/>
        <Property name="Title" value="D"/>
        <States>
          <State name="no">
            <AdditionalProperties/>
          </State>
          <State name="yes">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
      <Variable name="LungCancer" role="chance" type="finiteStates">
        <Coordinates x="421" y="152"/>
        <Property name="Relevance" value="7.0"/>
        <Property name="Title" value="L"/>
        <States>
          <State name="no">
            <AdditionalProperties/>
          </State>
          <State name="yes">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
      <Variable name="Smoker" role="chance" type="finiteStates">
        <Coordinates x="568" y="52"/>
        <Property name="Relevance" value="7.0"/>
        <Property name="Title" value="S"/>
        <States>
          <State name="no">
            <AdditionalProperties/>
          </State>
          <State name="yes">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
      <Variable name="Tuberculosis" role="chance" type="finiteStates">
        <Coordinates x="201" y="150"/>
        <Property name="Relevance" value="7.0"/>
        <Property name="Title" value="T"/>
        <States>
          <State name="no">
            <AdditionalProperties/>
          </State>
          <State name="yes">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
      <Variable name="TuberculosisOrCancer" role="chance" type="finiteStates">
        <Coordinates x="336" y="238"/>
        <Property name="Relevance" value="7.0"/>
        <Property name="Title" value="E"/>
        <States>
          <State name="no">
            <AdditionalProperties/>
          </State>
          <State name="yes">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
      <Variable name="VisitToAsia" role="chance" type="finiteStates">
        <Coordinates x="290" y="58"/>
        <Property name="Relevance" value="7.0"/>
        <Property name="Title" value="A"/>
        <States>
          <State name="no">
            <AdditionalProperties/>
          </State>
          <State name="yes">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
      <Variable name="X-ray" role="chance" type="finiteStates">
        <Comment>Indica si el test de rayos X ha sido positivo</Comment>
        <Coordinates x="252" y="322"/>
        <Property name="Relevance" value="7.0"/>
        <Property name="Title" value="X"/>
        <States>
          <State name="no">
            <AdditionalProperties/>
          </State>
          <State name="yes">
            <AdditionalProperties/>
          </State>
        </States>
      </Variable>
    </Variables>
    <Links>
      <Link directed="true" var1="Bronchitis" var2="Dyspnea">
        <AdditionalProperties/>
      </Link>
      <Link directed="true" var1="LungCancer" var2="TuberculosisOrCancer">
        <AdditionalProperties/>
      </Link>
      <Link directed="true" var1="Smoker" var2="Bronchitis">
        <AdditionalProperties/>
      </Link>
      <Link directed="true" var1="Smoker" var2="LungCancer">
        <AdditionalProperties/>
      </Link>
      <Link directed="true" var1="Tuberculosis" var2="TuberculosisOrCancer">
        <AdditionalProperties/>
      </Link>
      <Link directed="true" var1="TuberculosisOrCancer" var2="Dyspnea">
        <AdditionalProperties/>
      </Link>
      <Link directed="true" var1="TuberculosisOrCancer" var2="X-ray">
        <AdditionalProperties/>
      </Link>
      <Link directed="true" var1="VisitToAsia" var2="Tuberculosis">
        <AdditionalProperties/>
      </Link>
    </Links>
    <Potentials>
      <Potential role="conditionalProbability" type="Table">
        <Variables>
          <Variable name="X-ray"/>
          <Variable name="TuberculosisOrCancer"/>
        </Variables>
        <Values>0.95 0.05 0.02 0.98 </Values>
      </Potential>
      <Potential role="conditionalProbability" type="Table">
        <Variables>
          <Variable name="Bronchitis"/>
          <Variable name="Smoker"/>
        </Variables>
        <Values>0.7 0.3 0.4 0.6 </Values>
      </Potential>
      <Potential role="conditionalProbability" type="Table">
        <Variables>
          <Variable name="Dyspnea"/>
          <Variable name="Bronchitis"/>
          <Variable name="TuberculosisOrCancer"/>
        </Variables>
        <Values>0.9 0.1 0.3 0.7 0.2 0.8 0.1 0.9 </Values>
      </Potential>
      <Potential role="conditionalProbability" type="Table">
        <Variables>
          <Variable name="VisitToAsia"/>
        </Variables>
        <Values>0.99 0.01 </Values>
      </Potential>
      <Potential role="conditionalProbability" type="Table">
        <Variables>
          <Variable name="Smoker"/>
        </Variables>
        <Values>0.5 0.5 </Values>
      </Potential>
      <Potential role="conditionalProbability" type="Table">
        <Variables>
          <Variable name="LungCancer"/>
          <Variable name="Smoker"/>
        </Variables>
        <Values>0.99 0.01 0.9 0.1 </Values>
      </Potential>
      <Potential role="conditionalProbability" type="Table">
        <Variables>
          <Variable name="Tuberculosis"/>
          <Variable name="VisitToAsia"/>
        </Variables>
        <Values>0.99 0.01 0.95 0.05 </Values>
      </Potential>
      <Potential role="conditionalProbability" type="Table">
        <Variables>
          <Variable name="TuberculosisOrCancer"/>
          <Variable name="LungCancer"/>
          <Variable name="Tuberculosis"/>
        </Variables>
        <Values>1.0 0.0 0.0 1.0 0.0 1.0 0.0 1.0 </Values>
      </Potential>
    </Potentials>
    <AdditionalConstraints/>
    <AdditionalProperties/>
    <DecisionCriteria/>
  </ProbNet>
</ProbModelXML>""")
        self.assertDictEqual(model_data, model_data_expected)
        self.assertEqual(str(xmlfile.__str__()[:-1]), str(etree.tostring(xmlfile_expected)))
