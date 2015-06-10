#!/usr/bin/env python
import unittest
from io import StringIO
import networkx as nx
from pgmpy.readwrite import ProbModelXMLReader, ProbModelXMLWriter
import warnings
try:
    from lxml import etree
except ImportError:
    try:
        import xml.etree.cElementTree as etree
    except ImportError:
        try:
            import xml.etree.ElementTree as etree
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
            <Link var1="difficulty" var2="grade" directed="1">
                <Comment>Directed Edge from difficulty to grade</Comment>
                <Label>diff_to_grad</Label>
                <AdditionalProperties />
            </Link>
            <Link var1="intelligence" var2="grade" directed="1">
                <Comment>Directed Edge from intelligence to grade</Comment>
                <Label>intel_to_grad</Label>
                <AdditionalProperties />
            </Link>
            <Link var1="intelligence" var2="SAT" directed="1">
                <Comment>Directed Edge from intelligence to SAT</Comment>
                <Label>intel_to_sat</Label>
                <AdditionalProperties />
            </Link>
            <Link var1="grade" var2="recommendation_letter" directed="1">
                <Comment>Directed Edge from grade to recommendation_letter</Comment>
                <Label>grad_to_reco</Label>
                <AdditionalProperties />
            </Link>
        </Links>
        <Potential type="Table" role="ConditionalProbability" label="string">
            <Comment>CPDs in the form of table</Comment>
            <AdditionalProperties />
        </Potential>
    </ProbNet>
    <Policies />
    <InferenceOptions />
</ProbModelXML>
"""
        self.reader_string = ProbModelXMLReader(string=string)
        self.reader_file = ProbModelXMLReader(path=StringIO(string))

    def test_comment(self):
        comment_expected = "Student example model from Probabilistic Graphical Models: Principles and Techniques by Daphne Koller"
        self.maxDiff = None
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
        self.maxDiff = None
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
        self.maxDiff = None
        self.assertDictEqual(self.reader_string.probnet['edges'], edge)
        self.assertDictEqual(self.reader_file.probnet['edges'], edge)

    def test_additionalconstraints(self):
        additionalconstraints_expected = {'MaxNumParents':
                                          {'numParents': '5'}}
        self.maxDiff = None
        self.assertDictEqual(self.reader_string.probnet['AdditionalConstraints'],
                             additionalconstraints_expected)
        self.assertDictEqual(self.reader_file.probnet['AdditionalConstraints'],
                             additionalconstraints_expected)

    def test_additionalproperties(self):
        additionalproperties_expected = {'elvira.title': 'X ray result'}
        self.maxDiff = None
        self.assertDictEqual(self.reader_string.probnet['AdditionalProperties'],
                             additionalproperties_expected)
        self.assertDictEqual(self.reader_file.probnet['AdditionalProperties'],
                             additionalproperties_expected)


class TestProbModelXMLWriter(unittest.TestCase):
    def setUp(self):
        self.model_data = {'probnet':
                           {'type': 'BayesianNetwork',
                            'Language': 'English',
                            'AdditionalConstraints': {'MaxNumParents':
                                                      {'numParents': '5'}},
                            'AdditionalProperties': {'elvira.title': 'X ray result'},
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
                            'Comment': 'Student example model from Probabilistic Graphical Models: Principles and Techniques by Daphne Koller',
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
                                            'Variables': ['D0', 'D1', 'C0', 'C1'],
                                            'type': 'Tree/ADD',
                                            'UtilityVaribale': 'U1',
                                            'Branches': [{'Potential': {'type': 'Tree/ADD',
                                                                        'Branches': [{'Thresholds': [{'value': '-Infinity'},
                                                                                                     {'value': '0', 'belongsTo': 'Left'}],
                                                                                      'Potential': {'Subpotentials': [{'Potential': {'type': 'Table',
                                                                                                                                     'Values': '3'},
                                                                                                                       'type': 'Exponential'},
                                                                                                                      {'NumericVariables': ['C0', 'C1'],
                                                                                                                       'Potential': {'type': 'Table',
                                                                                                                                     'Values': '-1'},
                                                                                                                       'Coefficients': '4 -1',
                                                                                                                       'type': 'Exponential'}],
                                                                                                    'Variables': ['C0', 'C1'],
                                                                                                    'type': 'MixtureOfExponentials'}},
                                                                                     {'Thresholds': [{'value': '0', 'belongsTo': 'Left'},
                                                                                                     {'value': '+Infinity'}],
                                                                                      'Potential': {'Subpotentials': [{'NumericVariables': ['C1'],
                                                                                                                       'Potential': {'Variables': ['D1'],
                                                                                                                                     'type': 'Table',
                                                                                                                                     'Values': '10  5'},
                                                                                                                       'Coefficients': '0.25',
                                                                                                                       'type': 'Exponential'}],
                                                                                                    'Variables': ['C1', 'D1'],
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
                                                                        'Variables': ['C0'],
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
          <Variable name="C0"/>
          <Variable name="C1"/>
          <Variable name="D0"/>
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
</ProbModelXML>""")
        self.maxDiff = None
        self.assertEqual(str(self.writer.__str__()[:-1]), str(etree.tostring(self.expected_xml)))
