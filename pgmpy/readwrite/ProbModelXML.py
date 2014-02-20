"""
ProbModelXML: http://leo.ugr.es/pgm2012/submissions/pgm2012_submission_43.pdf

For the student example the ProbModelXML file should be:

<?xml version=“1.0” encoding=“UTF-8”?>
<ProbModelXML formatVersion=“1.0”>
    <ProbNet type=BayesianNetwork >
        <AdditionalConstraints />
        <Comment>
            Student example model from Probabilistic Graphical Models: Principles and Techniques by Daphne Koller
        </Comment>
        <Language>
            English
        </Language>
        <AdditionalProperties />
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
            <Variable name="grade" type="FiniteState" role="Chance">
                <Comment />
                <Coordinates />
                <AdditionalProperties />
                <States>
                    <State name="grade_A"><AdditionalProperties /></State>
                    <State name="grade_B"><AdditionalProperties /></State>
                    <State name="grade_C"><AdditionalProperties /></State>
                </States>
            </Variable>
            <Variable name="recommendation_letter" type="FiniteState" role="Chance">
                <Comment />
                <Coordinates />
                <AdditionalProperties />
                <States>
                    <State name="good"><AdditionalProperties /></State>
                    <State name="bad"><AdditionalProperties /></State>
                </States>
            </Variable>
            <Variable name="SAT" type="FiniteState" role="Chance">
                <Comment />
                <Coordinates />
                <AdditionalProperties />
                <States>
                    <State name="high"><AdditionalProperties /></State>
                    <State name="low"><AdditionalProperties /></State>
                </States>
            </Variable>
        </Variables>
        <Links>
            <Link var1="difficulty" var2="grade" directed=1>
                <Comment>Directed Edge from difficulty to grade</Comment>
                <Label>diff_to_grad</Label>
                <AdditionalProperties />
            </Link>
            <Link var1="intelligence" var2="grade" directed=1>
                <Comment>Directed Edge from intelligence to grade</Comment>
                <Label>intel_to_grad</Label>
                <AdditionalProperties />
            </Link>
            <Link var1="intelligence" var2="SAT" directed=1>
                <Comment>Directed Edge from intelligence to SAT</Comment>
                <Label>intel_to_sat</Label>
                <AdditionalProperties />
            </Link>
            <Link var1="grade" var2="recommendation_letter" directed=1>
                <Comment>Directed Edge from grade to recommendation_letter</Comment>
                <Label>grad_to_reco</Label>
                <AdditionalProperties />
            </Link>
        </Links>
        <Potential type="Table" role="ConditionalProbability" label=string>
            <Comment>CPDs in the form of table</Comment>
            <AdditionalProperties />
            <!--
                There is no specification in the paper about how the tables should be represented.
            -->
        </Potential>
    </ProbNet>
    <Policies />
    <InferenceOptions />
    <Evidence>
        <EvidenceCase>
            <Finding variable=string state=string stateIndex=integer numericValue=number/>
        </EvidenceCase>
    </Evidence>
</ProbModelXML>
"""
import warnings
import sys
import networkx as nx
try:
    from lxml import etree
except ImportError:
    try:
        import xml.etree.cElementTree as etree
    except ImportError:
        try:
            import xml.etree.ElementTree as etree
        except ImportError:
            print("Failed to import ElementTree from any known place")

warnings.warn("Not Complete. Please use only for reading and writing Bayesian Models only.")


class ProbModelXMLWriter(object):
    def __init__(self, network, encoding='utf-8', prettyprint=True,
                 language='English', comment=None):
        self.encoding = encoding
        self.prettyprint = prettyprint
        self.language = language
        self.comment = comment
        self.xml = etree.Element("ProbModelXML", attrib={'formatVersion': '1.0'})
        self.probnet = etree.SubElement(self.xml, 'ProbNet')
        self.variables = etree.SubElement(self.probnet, 'Variables')
        self.links = etree.SubElement(self.probnet, 'Links')
        self.potential = etree.SubElement(self.probnet, 'Potential')
        etree.SubElement(self.probnet, 'Comment').text = self.comment
        etree.SubElement(self.probnet, 'Language').text = self.language

        if isinstance(network, nx.DiGraph):
            self.probnet.attrib['type'] = 'BayesianNetwork'
            self.add_bayesian_network()
        elif isinstance(network, nx.Graph):
            self.probnet.attrib['type'] = 'MarkovNetwork'
            self.add_markov_network()

    def __str__(self):
        return etree.tostring(self.xml, encoding=self.encoding, prettyprint=self.prettyprint)

    def add_bayesian_network(self, network):
        for node in network.nodes():
            self.add_variables(node)
            self.add_potential(node)
        for edge in network.edges():
            self.add_links(edge)

    def add_policies(self):
        pass

    def add_inference_options(self):
        pass

    def add_evidence(self):
        pass

    def add_additional_constraints(self):
        pass

    def add_comment(self):
        pass

    def add_language(self):
        pass

    def add_additional_properties(self):
        pass

    def add_states(self):
        pass

    def add_variables(self, name, type='FiniteState', role='Chance', comment=None,
                      coordinates=None, states=None):
        variable = etree.SubElement(self.variables, 'Variable', attrib={'name': node,
                                                                        'type': 'FiniteState',
                                                                        'role': 'Chance'})
        etree.SubElement(variable, 'Comment').text = comment
        etree.SubElement(variable, 'Coordinates').text = coordinates
        etree.SubElement(variable, 'AdditionalProperties')
        state = variable.SubElement(variable, 'States')
        for s in states:
            etree.SubElement(state, 'State', attrib={'name': s}).append(etree.Element('AdditionalProperties'))

    def add_links(self, edge):
        pass

    def add_potential(self):
        pass

@open_file(1, mode='wb')
def write_probmodelxml(model, path, encoding='utf-8', prettyprint=True):
    pass