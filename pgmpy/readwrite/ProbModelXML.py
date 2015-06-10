"""
For the student example the ProbModelXML file should be:

<?xml version=“1.0” encoding=“UTF-8”?>
<ProbModelXML formatVersion=“1.0”>
    <ProbNet type="BayesianNetwork">
        <AdditionalConstraints />
        <Comment>
            Student example model from Probabilistic Graphical Models:
            Principles and Techniques by Daphne Koller
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
            <Variable name="recommendation_letter" type="FiniteState"
                    role="Chance">
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
                <Comment>Directed Edge from grade to
                    recommendation_letter</Comment>
                <Label>grad_to_reco</Label>
                <AdditionalProperties />
            </Link>
        </Links>
        <Potential type="Table" role="ConditionalProbability" label=string>
            <Comment>CPDs in the form of table</Comment>
            <AdditionalProperties />
            <!--
                There is no specification in the paper about
                how the tables should be represented.
            -->
        </Potential>
    </ProbNet>
    <Policies />
    <InferenceOptions />
    <Evidence>
        <EvidenceCase>
            <Finding variable=string state=string stateIndex=integer
                numericValue=number/>
        </EvidenceCase>
    </Evidence>
</ProbModelXML>
"""

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
            print("Failed to import ElementTree from any known place")

import networkx as nx

# warnings.warn("Not Complete. Please use only for "
#               "reading and writing Bayesian Models.")


def generate_probmodelxml(model, encoding='utf-8', prettyprint=True):
    """
    Generate ProbModelXML lines for model.

    Parameters
    ----------
    model : Graph
        The Bayesian or Markov Model
    encoding : string (optional)
        Encoding for text data
    prettyprint: bool (optional)
        If True uses line breaks and indenting in output XML.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> s = pgmpy.readwrite.generate_ProbModelXML(G) # doctest: +SKIP
    >>> for line in pgmpy.readwrite.generate_ProbModelXML(G):  #doctest: +SKIP
    ...     print(line)
    """
    writer = ProbModelXMLWriter(G, encoding=encoding, prettyprint=prettyprint)
    for line in str(writer).splitlines():
        yield line


# @open_file(1, mode='wb')
def write_probmodelxml(model, path, encoding='utf-8', prettyprint=True):
    """
    Write model in ProbModelXML format to path.

    Parameters
    ----------
    model : A NetworkX graph
            Bayesian network or Markov network
    path : file or string
            File or filename to write.
            Filenames ending in .gz or .bz2 will be compressed.
    encoding : string (optional)
            Encoding for text data.
    prettyprint : bool (optional)
            If True use line breaks and indenting in output XML.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pgmpy.readwrite.write_probmodelxml(G, "test.probmodelxml")
    """
    writer = ProbModelXMLWriter(model, path, encoding=encoding,
                                prettyprint=prettyprint)
    writer.dump(path)


# @open_file(0, mode='rb')
def read_probmodelxml(path):
    """
    Read model in ProbModelXML format from path.

    Parameters
    ----------
    path : file or string
        file or filename from which to read.

    Returns
    -------
    model : NetworkX Graph
            A BayesianModel or MarkovModel object depending on the
            type of model the XML represents.

    Examples
    --------
    >>> G = pgmpy.readwrite.read_probmodelxml('test.probModelXML')
    """
    reader = ProbModelXMLReader(path=path)
    return reader.make_network()


def parse_probmodelxml(string):
    """
    Read model in ProbModelXML format from string.

    Parameters
    ----------
    string : string
        String containing ProbModelXML information.
        (e.g., contents of a ProbModelXML file).

    Returns
    -------
    model : NetworkX Graph
        A BayesianModel or MarkovModel object depending on the XML given.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> linefeed = chr(10)
    >>> s = linefeed.join(pgmpy.readwrite.generate_probmodelxml(G))
    >>> H = pgmpy.readwrite.parse_probmodelxml(s)
    """
    reader = ProbModelXMLReader(string=string)
    return reader.make_network()


class ProbModelXMLWriter:
    """
    Class for writing models in ProbModelXML format.
    """
    def __init__(self, model_data, encoding='utf-8', prettyprint=True):
        """
        Initialize a ProbModelXMLWriter Object.

        Parameters
        ----------
        model : A BayesianModel or MarkovModel
            The model to write.
        encoding : string (optional)
            Encoding for text data
        prettyprint : bool (optional)
            If True uses line breaks and indentation in output XML.

        Examples
        --------

        Refernces
        ---------
        [1] http://leo.ugr.es/pgm2012/submissions/pgm2012_submission_43.pdf
        [2] http://www.cisiad.uned.es/techreports/ProbModelXML.pdf
        """
        # TODO: add policies, InferenceOptions, Evidence
        # TODO: add parsing of language and comments and additional properties
        self.data = model_data
        self.encoding = encoding
        self.prettyprint = prettyprint

        # Creating initial tags
        self.xml = etree.Element("ProbModelXML", attrib={'formatVersion': '1.0'})
        self.probnet = etree.SubElement(self.xml, 'ProbNet')
        self.variables = etree.SubElement(self.probnet, 'Variables')
        self.links = etree.SubElement(self.probnet, 'Links')
        self.potentials = etree.SubElement(self.probnet, 'Potentials')
        self.additional_constraints = etree.SubElement(self.probnet, 'AdditionalConstraints')

        # adding information for probnet
        self.probnet.attrib['type'] = self.data['probnet']['type']
        try:
            etree.SubElement(self.probnet, 'Language').text = self.data['probnet']['Language']
        except KeyError:
            pass
        try:
            etree.SubElement(self.probnet, 'Comment').text = self.data['probnet']['Comment']
        except KeyError:
            pass
        try:
            self._add_additional_properties(self.xml, self.data['probnet']['AdditionalProperties'])
        except KeyError:
            etree.SubElement(self.probnet, 'AdditionalProperties')

        # Add Additional Constraints
        for constraint in sorted(self.data['probnet']['AdditionalConstraints']):
            self._add_constraint(constraint)

        # Add variables
        for variable in sorted(self.data['probnet']['Variables']):
            self._add_variable(variable)

        # Add edges
        for edge in sorted(self.data['probnet']['edges']):
            self._add_link(edge)

        # Add Potentials
        for potential in sorted(self.data['probnet']['Potentials']):
            self._add_potential(potential, self.potentials)

    def __str__(self):
        """
        Return the XML as string.
        """
        if self.prettyprint:
            self.indent(self.xml)
        return etree.tostring(self.xml, encoding=self.encoding)

    @staticmethod
    def _add_additional_properties(position, properties_dict):
        """
        Sets AdditionalProperties of the ProbModelXML.
        """
        add_prop = etree.SubElement(position, 'AdditionalProperties')
        for key, value in properties_dict.items():
            etree.SubElement(add_prop, 'Property', attrib={'name': key, 'value': value})

    def _add_variable(self, variable):
        """
        Adds a node to the ProbModelXML.
        """
        # TODO: Add feature for accepting additional properties of states.
        variable_data = self.data['probnet']['Variables'][variable]
        variable_element = etree.SubElement(self.variables, 'Variable', attrib={'name': variable,
                                                                                'type': variable_data['type'],
                                                                                'role': variable_data['role']})
        etree.SubElement(variable_element, 'Comment').text = variable_data['Comment']
        etree.SubElement(variable_element, 'Coordinates', variable_data['Coordinates'])
        try:
            for key, value in variable_data['AdditionalProperties'].items():
                etree.SubElement(variable_element, 'Property', attrib={'name': key, 'value': value})
        except KeyError:
            etree.SubElement(variable_element, 'AdditionalProperties')
        states = etree.SubElement(variable_element, 'States')
        for s in sorted(variable_data['States']):
            state = etree.SubElement(states, 'State', attrib={'name': s})
            try:
                self._add_additional_properties(state, variable_data['States'][s]['AdditionalProperties'])
            except KeyError:
                etree.SubElement(state, 'AdditionalProperties')

    def _add_link(self, edge):
        """
        Adds an edge to the ProbModelXML.
        """
        edge_data = self.data['probnet']['edges'][edge]
        link = etree.SubElement(self.links, 'Link', attrib={'var1': edge[0], 'var2': edge[1],
                                                            'directed': edge_data['directed']})
        etree.SubElement(link, 'Comment').text = edge_data['Comment']
        etree.SubElement(link, 'Label').text = edge_data['Label']
        try:
            self._add_additional_properties(link, edge_data['AdditionalProperties'])
        except KeyError:
            etree.SubElement(link, 'AdditionalProperties')

    def _add_constraint(self, constraint):
        """
        Adds constraint to the ProbModelXML.
        """
        constraint_data = self.data['probnet']['AdditionalConstraints'][constraint]
        constraint_element = etree.SubElement(
            self.additional_constraints, 'Constraint', attrib={'name': constraint})
        for argument in sorted(constraint_data):
            name = argument
            value = constraint_data[name]
            etree.SubElement(constraint_element, 'Argument', attrib={'name': name, 'value': value})

    def _add_decision_criteria(self, criteria_dict):
        """
        Adds Decision Criteria to the ProbModelXML.

        Parameters
        ----------
        criteria_dict: dict
            Dictionary containing Deecision Criteria data.
            For example: {'effectiveness': {}, 'cost': {}}

        Examples
        -------
        >>> writer = ProbModelXMLWriter(model)
        >>> writer._add_decision_criteria(criteria_dict)
        """
        decision_tag = etree.SubElement(self.xml, 'DecisionCriteria', attrib={})
        for criteria in sorted(criteria_dict):
            criteria_tag = etree.SubElement(decision_tag, 'Criterion', attrib={'name': criteria})
            self._add_additional_properties(criteria_tag, criteria_dict[criteria])

    def _add_potential(self, potential, parent_tag):
        """
        Adds Potentials to the ProbModelXML.

        Parameters
        ----------
        potential: dict
            Dictionary containing Potential data.
            For example: {'role': 'Utility',
                          'Variables': ['D0', 'D1', 'C0', 'C1'],
                          'type': 'Tree/ADD',
                          'UtilityVaribale': 'U1'}
        parent_tag: etree Element
            etree element which would contain potential tag
            For example: <Element Potentials at 0x7f315fc44b08>
                         <Element Branch at 0x7f315fc44c88>
                         <Element Branch at 0x7f315fc44d88>
                         <Element Subpotentials at 0x7f315fc44e48>

        Examples
        -------
        >>> writer = ProbModelXMLWriter(model)
        >>> writer._add_potential(potential, parent_tag)
        """
        potential_type = potential['type']
        try:
            potential_tag = etree.SubElement(parent_tag, 'Potential', attrib={
                'type': potential['type'], 'role': potential['role']})
        except KeyError:
            potential_tag = etree.SubElement(parent_tag, 'Potential', attrib={
                'type': potential['type']})
        if 'Comment' in potential:
            etree.SubElement(potential_tag, 'Comment').text = potential['Comment']
        if 'AdditionalProperties' in potential:
            self._add_additional_properties(potential_tag, potential['AdditionalProperties'])
        if potential_type == "delta":
            etree.SubElement(potential_tag, 'Variable', attrib={'name': potential['Variable']})
            if 'State' in potential:
                etree.SubElement(potential_tag, 'State').text = potential['State']
            if 'StateIndex' in potential:
                etree.SubElement(potential_tag, 'StateIndex').text = potential['StateIndex']
            if 'NumericValue' in potential:
                etree.SubElement(potential_tag, 'NumericValue').text = potential['NumericValue']
        else:
            if 'UtilityVariable' in potential:
                etree.SubElement(potential_tag, 'UtilityVariable', attrib={
                    'name': potential['UtilityVariable']})
            if 'Variables' in potential:
                variable_tag = etree.SubElement(potential_tag, 'Variables')
                for var in sorted(potential['Variables']):
                    etree.SubElement(variable_tag, 'Variable', attrib={'name': var})
            if 'Values' in potential:
                etree.SubElement(potential_tag, 'Values').text = potential['Values']
            if 'UncertainValues' in potential:
                value_tag = etree.SubElement(potential_tag, 'UncertainValues', attrib={})
                for value in sorted(potential['UncertainValues']):
                    try:
                        etree.SubElement(value_tag, 'Value', attrib={
                            'distribution': value['distribution'],
                            'name': value['name']}).text = value['value']
                    except KeyError:
                        etree.SubElement(value_tag, 'Value', attrib={
                            'distribution': value['distribution']}).text = value['value']
            if 'TopVariable' in potential:
                etree.SubElement(potential_tag, 'TopVariable', attrib={'name': potential['TopVariable']})
            if 'Branches' in potential:
                branches_tag = etree.SubElement(potential_tag, 'Branches')
                for branch in potential['Branches']:
                    branch_tag = etree.SubElement(branches_tag, 'Branch')
                    if 'States' in branch:
                        states_tag = etree.SubElement(branch_tag, 'States')
                        for state in sorted(branch['States']):
                            etree.SubElement(states_tag, 'State', attrib={'name': state['name']})
                    if 'Potential' in branch:
                        self._add_potential(branch['Potential'], branch_tag)
                    if 'Label' in branch:
                        etree.SubElement(branch_tag, 'Label').text = branch['Label']
                    if 'Reference' in branch:
                        etree.SubElement(branch_tag, 'Reference').text = branch['Reference']
                    if 'Thresholds' in branch:
                        thresholds_tag = etree.SubElement(branch_tag, 'Thresholds')
                        for threshold in branch['Thresholds']:
                            try:
                                etree.SubElement(thresholds_tag, 'Threshold', attrib={
                                    'value': threshold['value'], 'belongsTo': threshold['belongsTo']})
                            except KeyError:
                                etree.SubElement(thresholds_tag, 'Threshold', attrib={
                                    'value': threshold['value']})
            if 'Model' in potential:
                etree.SubElement(potential_tag, 'Model').text = potential['Model']
            if 'Subpotentials' in potential:
                subpotentials = etree.SubElement(potential_tag, 'Subpotentials')
                for subpotential in potential['Subpotentials']:
                    self._add_potential(subpotential, subpotentials)
            if 'Coefficients' in potential:
                etree.SubElement(potential_tag, 'Coefficients').text = potential['Coefficients']
            if 'CovarianceMatrix' in potential:
                etree.SubElement(potential_tag, 'CovarianceMatrix').text = potential['CovarianceMatrix']
            if 'Potential' in potential:
                self._add_potential(potential['Potential'], potential_tag)
            if 'NumericVariables' in potential:
                numvar_tag = etree.SubElement(potential_tag, 'NumericVariables')
                for var in sorted(potential['NumericVariables']):
                    etree.SubElement(numvar_tag, 'Variable', attrib={'name': var})

    def dump(self, stream):
        """
        Dumps the data to stream after appending header.
        """
        if self.prettyprint:
            self.indent(self.xml)
        document = etree.ElementTree(self.xml)
        header = '<?xml version="1.0" encoding="%s"?>' % self.encoding
        stream.write(header.encode(self.encoding))
        document.write(stream, encoding=self.encoding)

    def indent(self, elem, level=0):
        """
        Inplace prettyprint formatter.
        """
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


class ProbModelXMLReader:
    """
    Class for reading ProbModelXML format from files or strings.
    """
    # TODO: add methods to parse policies, inferenceoption, evidence etc.
    # TODO: add reading formatVersion
    def __init__(self, path=None, string=None):
        """
        Initialize an instance of ProbModelXMLReader class.

        Parameters
        ----------
        path : file or string
            File containing ProbModelXML information.
        string : string
            String containing ProbModelXML information.

        Example
        -------
        >>> reader = ProbModelXMLReader('test.ProbModelXML')

        Structure of Probnet Object
        ---------------------------
        { probnet: { type:
                     Comment:
                     Language:
                     AdditionalProperties: { property_name1: property_value1,
                                           property_name2: property_value2,
                                                        ....
                                         }
                     Variables: { variable_name1: { type:
                                                    roles:
                                                    Comment:
                                                    Coordinates:
                                                    AdditionalProperties: { property_name1: property_value1,
                                                                            property_name2: property_value2,
                                                                                        ....
                                                                          }
                                                    states: { state1: {AdditionalProperties: {
                                                                                               ....
                                                                                               ....
                                                                                             }
                                                              state2: {AdditionalProperties: {
                                                                                               ....
                                                                                               ....
                                                                                             }
                                                                 .....
                                                            }
                                                  }
                                  variable_name2: {
                                                    ...........
                                                  }
                                      .........
                                 }
                     edges: { (var1, var2): { directed:
                                              Comment:
                                              Label:
                                              AdditionalProperties: { property_name1: property_value1,
                                                                      property_name2: property_value2,
                                                                                 .....
                                                                    }
                              (var3, var4): {
                                              .....
                                              .....
                                            }
                                   ........
                            }
                    }
        }

        References
        ----------
        [1] http://leo.ugr.es/pgm2012/submissions/pgm2012_submission_43.pdf
        [2] http://www.cisiad.uned.es/techreports/ProbModelXML.pdf
        """
        if path is not None:
            self.xml = etree.ElementTree(file=path)
        elif string is not None:
            self.xml = etree.fromstring(string)
        else:
            raise ValueError("Must specify either 'path' or 'string' as kwarg.")

        self.create_probnet()

    def create_probnet(self):
        """
        Returns a BayesianModel or MarkovModel object depending on the
        type of ProbModelXML passed to ProbModelXMLReader class.
        """
        self.probnet = {}
        # Add general properties
        probnet_elem = self.xml.find('ProbNet')
        self.probnet['type'] = probnet_elem.attrib['type']
        self.add_comment(probnet_elem.find('Comment').text)
        self.add_language(probnet_elem.find('Language').text)
        if probnet_elem.find('AdditionalProperties') is not None:
            self.probnet['AdditionalProperties'] = {}
            for prop in probnet_elem.find('AdditionalProperties'):
                self.add_additional_property(self.probnet['AdditionalProperties'], prop)

        # Add additional Constraints
        self.probnet['AdditionalConstraints'] = {}
        for constraint in probnet_elem.findall('AdditionalConstraints/Constraint'):
            self.add_probnet_additionalconstraints(constraint)

        # Add nodes
        self.probnet['Variables'] = {}
        for variable in probnet_elem.find('Variables'):
            self.add_node(variable)

        # Add edges
        self.probnet['edges'] = {}
        for edge in self.xml.findall('.//Links')[0]:
            self.add_edge(edge)

        # Add CPD
        for potential in self.xml.findall('.//Potential')[0]:
            self.add_potential(potential)

    def add_probnet_additionalconstraints(self, constraint):
        constraint_name = constraint.attrib['name']
        self.probnet['AdditionalConstraints'][constraint_name] = {}
        for argument in constraint.findall('Argument'):
            argument_name = argument.attrib['name']
            argument_value = argument.attrib['value']
            self.probnet['AdditionalConstraints'][constraint_name][argument_name] = argument_value

    def add_comment(self, comment):
        self.probnet['Comment'] = comment

    def add_language(self, language):
        self.probnet['Language'] = language

    @staticmethod
    def add_additional_property(place, prop):
        place[prop.attrib['name']] = prop.attrib['value']

    def add_node(self, variable):
        # TODO: Do some checks with variable type and roles. Right now I don't know when they are to be used.
        variable_name = variable.attrib['name']
        self.probnet['Variables'][variable_name] = {}
        self.probnet['Variables'][variable_name]['type'] = variable.attrib['type']
        self.probnet['Variables'][variable_name]['role'] = variable.attrib['role']
        if variable.find('Comment') is not None:
            self.probnet['Variables'][variable_name]['Comment'] = variable.find('Comment').text
        if variable.find('Coordinates') is not None:
            self.probnet['Variables'][variable_name]['Coordinates'] = variable.find('Coordinates').attrib
        if variable.find('AdditionalProperties/Property') is not None:
            for prop in variable.findall('AdditionalProperties/Property'):
                self.probnet['Variables'][variable_name]['AdditionalProperties'][prop.attrib['name']] = \
                    prop.attrib['value']
        if variable.find('States/State') is None:
            warnings.warn("States not available for node: " + variable_name)
        else:
            self.probnet['Variables'][variable_name]['States'] = {state.attrib['name']: {prop.attrib['name']: prop.attrib['value'] for prop in state.findall('AdditionalProperties/Property')} for state in variable.findall('States/State')}

    def add_edge(self, edge):
        var1 = edge.attrib['var1']
        var2 = edge.attrib['var2']
        self.probnet['edges'][(var1, var2)] = {}
        self.probnet['edges'][(var1, var2)]['directed'] = edge.attrib['directed']
        # TODO: check for the case of undirected graphs if we need to add to both elements of the dic for a single edge.
        if edge.find('Comment') is not None:
            self.probnet['edges'][(var1, var2)]['Comment'] = edge.find('Comment').text
        if edge.find('Label') is not None:
            self.probnet['edges'][(var1, var2)]['Label'] = edge.find('Label').text
        if edge.find('AdditionalProperties/Property') is not None:
            for prop in edge.findall('AdditionalProperties/Property'):
                self.probnet['edges'][(var1, var2)]['AdditionalProperties'][prop.attrib['name']] = prop.attrib['value']

    def add_potential(self, potential):
        # TODO: Add code to read potential
        pass
