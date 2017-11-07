#!/usr/bin/env python


from io import BytesIO
import pyparsing as pp


# TODO input and output state


try:
    from lxml import etree
except ImportError:
    try:
        import xml.etree.ElementTree as etree
    except ImportError:
        # try:
        #    import xml.etree.cElementTree as etree
        #    commented out because xml.etree.cElementTree is giving errors with dictionary attributes
        print("Failed to import ElementTree from any known place")

import numpy as np

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD, State
from pgmpy.extern.six.moves import map


class XMLBIFReader(object):
    """
    Base class for reading network file in XMLBIF format.
    """

    def __init__(self, path=None, string=None):
        """
        Initialisation of XMLBIFReader object.

        Parameters
        ----------
        path : file or str
            File of XMLBIF data
            File of XMLBIF data
        string : str
            String of XMLBIF data

        Examples
        --------
        # xmlbif_test.xml is the file present in
        # http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        """
        if path:
            self.network = etree.ElementTree(file=path).getroot().find('NETWORK')
        elif string:
            self.network = etree.fromstring(string.encode('utf-8')).find('NETWORK')
        else:
            raise ValueError("Must specify either path or string")
        self.network_name = self.network.find('NAME').text
        self.variables = self.get_variables()
        self.variable_parents = self.get_parents()
        self.edge_list = self.get_edges()
        self.variable_states = self.get_states()
        self.variable_CPD = self.get_values()
        self.variable_property = self.get_property()

    def get_variables(self):
        """
        Returns list of variables of the network

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_variables()
        ['light-on', 'bowel-problem', 'dog-out', 'hear-bark', 'family-out']
        """
        variables = [variable.find('NAME').text for variable in self.network.findall('VARIABLE')]
        return variables

    def get_edges(self):
        """
        Returns the edges of the network

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_edges()
        [['family-out', 'light-on'],
         ['family-out', 'dog-out'],
         ['bowel-problem', 'dog-out'],
         ['dog-out', 'hear-bark']]
        """
        edge_list = [[value, key] for key in self.variable_parents
                     for value in self.variable_parents[key]]
        return edge_list

    def get_states(self):
        """
        Returns the states of variables present in the network

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_states()
        {'bowel-problem': ['true', 'false'],
         'dog-out': ['true', 'false'],
         'family-out': ['true', 'false'],
         'hear-bark': ['true', 'false'],
         'light-on': ['true', 'false']}
        """
        variable_states = {variable.find('NAME').text: [outcome.text for outcome in variable.findall('OUTCOME')]
                           for variable in self.network.findall('VARIABLE')}
        return variable_states

    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_parents()
        {'bowel-problem': [],
         'dog-out': ['family-out', 'bowel-problem'],
         'family-out': [],
         'hear-bark': ['dog-out'],
         'light-on': ['family-out']}
        """
        variable_parents = {definition.find('FOR').text: [edge.text for edge in definition.findall('GIVEN')]
                            for definition in self.network.findall('DEFINITION')}
        return variable_parents

    def get_values(self):
        """
        Returns the CPD of the variables present in the network

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_values()
        {'bowel-problem': array([[ 0.01],
                                 [ 0.99]]),
         'dog-out': array([[ 0.99,  0.01,  0.97,  0.03],
                           [ 0.9 ,  0.1 ,  0.3 ,  0.7 ]]),
         'family-out': array([[ 0.15],
                              [ 0.85]]),
         'hear-bark': array([[ 0.7 ,  0.3 ],
                             [ 0.01,  0.99]]),
         'light-on': array([[ 0.6 ,  0.4 ],
                            [ 0.05,  0.95]])}
        """
        variable_CPD = {definition.find('FOR').text: list(map(float, table.text.split()))
                        for definition in self.network.findall('DEFINITION')
                        for table in definition.findall('TABLE')}
        for variable in variable_CPD:
            arr = np.array(variable_CPD[variable])
            arr = arr.reshape((len(self.variable_states[variable]),
                               arr.size // len(self.variable_states[variable])), order='F')
            variable_CPD[variable] = arr
        return variable_CPD

    def get_property(self):
        """
        Returns the property of the variable

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_property()
        {'bowel-problem': ['position = (190, 69)'],
         'dog-out': ['position = (155, 165)'],
         'family-out': ['position = (112, 69)'],
         'hear-bark': ['position = (154, 241)'],
         'light-on': ['position = (73, 165)']}
        """
        variable_property = {variable.find('NAME').text: [property.text for property in variable.findall('PROPERTY')]
                             for variable in self.network.findall('VARIABLE')}
        return variable_property

    def get_model(self):
        model = BayesianModel()
        model.add_nodes_from(self.variables)
        model.add_edges_from(self.edge_list)
        model.name = self.network_name

        tabular_cpds = []
        for var, values in self.variable_CPD.items():
            evidence_card = [len(self.variable_states[evidence_var]) for evidence_var in self.variable_parents[var]]
            cpd = TabularCPD(var, len(self.variable_states[var]), values,
                             evidence=self.variable_parents[var],
                             evidence_card=evidence_card, state_names=self.get_states())
            tabular_cpds.append(cpd)

        model.add_cpds(*tabular_cpds)

        for node, properties in self.variable_property.items():
            for prop in properties:
                if prop is not None:
                    prop_name, prop_value = map(lambda t: t.strip(), prop.split('='))
                    model.node[node][prop_name] = prop_value

        return model


class XMLBIFWriter(object):
    """
    Base class for writing XMLBIF network file format.
    """

    def __init__(self, model, encoding='utf-8', prettyprint=True):
        """
        Initialise a XMLBIFWriter object.

        Parameters
        ----------
        model: BayesianModel Instance
            Model to write
        encoding: str (optional)
            Encoding for text data
        prettyprint: Bool(optional)
            Indentation in output XML if true

        Examples
        --------
        >>> writer = XMLBIFWriter(model)
        """
        if not isinstance(model, BayesianModel):
            raise TypeError("model must an instance of BayesianModel")
        self.model = model

        self.encoding = encoding
        self.prettyprint = prettyprint

        self.xml = etree.Element("BIF", attrib={'VERSION': '0.3'})
        self.network = etree.SubElement(self.xml, 'NETWORK')
        if self.model.name:
            etree.SubElement(self.network, 'NAME').text = self.model.name
        else:
            etree.SubElement(self.network, 'NAME').text = "UNTITLED"

        self.variables = self.get_variables()
        self.states = self.get_states()
        self.properties = self.get_properties()
        self.definition = self.get_definition()
        self.tables = self.get_values()

    def __str__(self):
        """
        Return the XML as string.
        """
        if self.prettyprint:
            self.indent(self.xml)
        f = BytesIO()
        et = etree.ElementTree(self.xml)
        et.write(f, encoding=self.encoding, xml_declaration=True)
        return f.getvalue().decode(self.encoding)

    def indent(self, elem, level=0):
        """
        Inplace prettyprint formatter.
        """
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def get_variables(self):
        """
        Add variables to XMLBIF

        Return
        ------
        dict: dict of type {variable: variable tags}

        Examples
        --------
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_variables()
        {'bowel-problem': <Element VARIABLE at 0x7fe28607dd88>,
         'family-out': <Element VARIABLE at 0x7fe28607de08>,
         'hear-bark': <Element VARIABLE at 0x7fe28607de48>,
         'dog-out': <Element VARIABLE at 0x7fe28607ddc8>,
         'light-on': <Element VARIABLE at 0x7fe28607de88>}
        """
        variables = self.model.nodes()
        variable_tag = {}
        for var in sorted(variables):
            variable_tag[var] = etree.SubElement(self.network, "VARIABLE", attrib={'TYPE': 'nature'})
            etree.SubElement(variable_tag[var], "NAME").text = var
        return variable_tag

    def get_states(self):
        """
        Add outcome to variables of XMLBIF

        Return
        ------
        dict: dict of type {variable: outcome tags}

        Examples
        --------
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_states()
        {'dog-out': [<Element OUTCOME at 0x7ffbabfcdec8>, <Element OUTCOME at 0x7ffbabfcdf08>],
         'family-out': [<Element OUTCOME at 0x7ffbabfd4108>, <Element OUTCOME at 0x7ffbabfd4148>],
         'bowel-problem': [<Element OUTCOME at 0x7ffbabfd4088>, <Element OUTCOME at 0x7ffbabfd40c8>],
         'hear-bark': [<Element OUTCOME at 0x7ffbabfcdf48>, <Element OUTCOME at 0x7ffbabfcdf88>],
         'light-on': [<Element OUTCOME at 0x7ffbabfcdfc8>, <Element OUTCOME at 0x7ffbabfd4048>]}
        """
        outcome_tag = {}
        cpds = self.model.get_cpds()
        for cpd in cpds:
            var = cpd.variable
            outcome_tag[var] = []
            if cpd.state_names is None or cpd.state_names.get(var) is None:
                states = range(cpd.get_cardinality([var])[var])
            else:
                states = cpd.state_names[var]

            for state in states:
                state_tag = etree.SubElement(self.variables[var], "OUTCOME")
                state_tag.text = self._make_valid_state_name(state)
                outcome_tag[var].append(state_tag)
        return outcome_tag

    def _make_valid_state_name(self, state_name):
        """Transform the input state_name into a valid state in XMLBIF.
        XMLBIF states must start with a letter an only contain letters,
        numbers and underscores.
        """
        s = str(state_name)
        s_fixed = pp.CharsNotIn(pp.alphanums + "_").setParseAction(pp.replaceWith("_")).transformString(s)
        if not s_fixed[0].isalpha():
            s_fixed = "state" + s_fixed
        return s_fixed

    def get_properties(self):
        """
        Add property to variables in XMLBIF

        Return
        ------
        dict: dict of type {variable: property tag}

        Examples
        --------
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_property()
        {'light-on': <Element PROPERTY at 0x7f7a2ffac1c8>,
         'family-out': <Element PROPERTY at 0x7f7a2ffac148>,
         'hear-bark': <Element PROPERTY at 0x7f7a2ffac188>,
         'bowel-problem': <Element PROPERTY at 0x7f7a2ffac0c8>,
         'dog-out': <Element PROPERTY at 0x7f7a2ffac108>}
        """
        variables = self.model.nodes()
        property_tag = {}
        for var in sorted(variables):
            properties = self.model.node[var]
            property_tag[var] = etree.SubElement(self.variables[var], "PROPERTY")
            for prop, val in properties.items():
                property_tag[var].text = str(prop) + " = " + str(val)
        return property_tag

    def get_definition(self):
        """
        Add Definition to XMLBIF

        Return
        ------
        dict: dict of type {variable: definition tag}

        Examples
        --------
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_definition()
        {'hear-bark': <Element DEFINITION at 0x7f1d48977408>,
         'family-out': <Element DEFINITION at 0x7f1d489773c8>,
         'dog-out': <Element DEFINITION at 0x7f1d48977388>,
         'bowel-problem': <Element DEFINITION at 0x7f1d48977348>,
         'light-on': <Element DEFINITION at 0x7f1d48977448>}
        """
        cpds = self.model.get_cpds()
        cpds.sort(key=lambda x: x.variable)
        definition_tag = {}
        for cpd in cpds:
            definition_tag[cpd.variable] = etree.SubElement(self.network, "DEFINITION")
            etree.SubElement(definition_tag[cpd.variable], "FOR").text = cpd.variable
            for child in sorted(cpd.variables[:0:-1]):
                etree.SubElement(definition_tag[cpd.variable], "GIVEN").text = child

        return definition_tag

    def get_values(self):
        """
        Add Table to XMLBIF.

        Return
        ---------------
        dict: dict of type {variable: table tag}

        Examples
        -------
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_values()
        {'dog-out': <Element TABLE at 0x7f240726f3c8>,
         'light-on': <Element TABLE at 0x7f240726f488>,
         'bowel-problem': <Element TABLE at 0x7f240726f388>,
         'family-out': <Element TABLE at 0x7f240726f408>,
         'hear-bark': <Element TABLE at 0x7f240726f448>}
        """
        cpds = self.model.get_cpds()
        definition_tag = self.definition
        table_tag = {}
        for cpd in cpds:
            table_tag[cpd.variable] = etree.SubElement(definition_tag[cpd.variable], "TABLE")
            table_tag[cpd.variable].text = ''
            for val in cpd.get_values().ravel(order="F"):
                table_tag[cpd.variable].text += str(val) + ' '

        return table_tag

    def write_xmlbif(self, filename):
        """
        Write the xml data into the file.

        Parameters
        ----------
        filename: Name of the file.

        Examples
        -------
        >>> writer = XMLBIFWriter(model)
        >>> writer.write_xmlbif(test_file)
        """
        with open(filename, 'w') as fout:
            fout.write(self.__str__())
