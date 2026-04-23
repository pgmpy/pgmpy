#!/usr/bin/env python

import warnings
import xml.etree.ElementTree as etree
from io import BytesIO
from itertools import chain

import numpy as np

from pgmpy import logger
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import compat_fns

try:
    import pyparsing as pp
except ImportError as e:
    raise ImportError(
        f"{e} . pyparsing is required for using read/write methods. Please install using: pip install pyparsing."
    ) from None


class XMLBIFReader:
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
    >>> # xmlbif_test.xml is the file present in
    >>> # http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
    >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/asia")
    >>> writer = XMLBIFWriter(model)
    >>> writer.write("xmlbif_test.xml")
    >>> reader = XMLBIFReader("xmlbif_test.xml")
    >>> model = reader.get_model()

    Reference
    ---------
    [1] https://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/
    """

    def __init__(self, path=None, string=None):
        if path:
            self.network = etree.ElementTree(file=path).getroot().find("NETWORK")
        elif string:
            self.network = etree.fromstring(string.encode("utf-8")).find("NETWORK")
        else:
            raise ValueError("Must specify either path or string")
        self.network_name = self.network.find("NAME").text
        self.variables = self.get_variables()
        self.variable_parents = self.get_parents()
        self.edge_list = self.get_edges()
        self.variable_states = self.get_states()
        self.variable_CPD = self.get_values()
        self.variable_property = self.get_property()
        self.state_names = self.get_states()

    def get_variables(self):
        """
        Returns list of variables of the network

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> sorted(reader.get_variables())
        ['asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray']
        """
        variables = [variable.find("NAME").text for variable in self.network.findall("VARIABLE")]
        return variables

    def get_edges(self):
        """
        Returns the edges of the network

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_edges() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [['smoke', 'bronc'], ['bronc', 'dysp'],
        ['either', 'dysp'], ['lung', 'either'],
        ['tub', 'either'], ['smoke', 'lung'],
        ['asia', 'tub'], ['either', 'xray']]
        """
        edge_list = [[value, key] for key in self.variable_parents for value in self.variable_parents[key]]
        return edge_list

    def get_states(self):
        """
        Returns the states of variables present in the network

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_states() # doctest: +NORMALIZE_WHITESPACE
        {'asia': ['yes', 'no'],
        'bronc': ['yes', 'no'],
        'dysp': ['yes', 'no'],
        'either': ['yes', 'no'],
        'lung': ['yes', 'no'],
        'smoke': ['yes', 'no'],
        'tub': ['yes', 'no'],
        'xray': ['yes', 'no']}
        """
        variable_states = {
            variable.find("NAME").text: [outcome.text for outcome in variable.findall("OUTCOME")]
            for variable in self.network.findall("VARIABLE")
        }
        return variable_states

    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_parents() # doctest: +NORMALIZE_WHITESPACE
        {'asia': [],
        'bronc': ['smoke'],
        'dysp': ['bronc', 'either'],
        'either': ['lung', 'tub'],
        'lung': ['smoke'],
        'smoke': [],
        'tub': ['asia'],
        'xray': ['either']}
        """
        variable_parents = {
            definition.find("FOR").text: [edge.text for edge in definition.findall("GIVEN")]
            for definition in self.network.findall("DEFINITION")
        }
        return variable_parents

    def get_values(self):
        """
        Returns the CPD of the variables present in the network

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_values() # doctest: +NORMALIZE_WHITESPACE
        {'asia': array([[0.01],
           [0.99]]), 'bronc': array([[0.6, 0.3],
           [0.4, 0.7]]), 'dysp': array([[0.9, 0.8, 0.7, 0.1],
           [0.1, 0.2, 0.3, 0.9]]), 'either': array([[1., 1., 1., 0.],
           [0., 0., 0., 1.]]), 'lung': array([[0.1 , 0.01],
           [0.9 , 0.99]]), 'smoke': array([[0.5],
           [0.5]]), 'tub': array([[0.05, 0.01],
           [0.95, 0.99]]), 'xray': array([[0.98, 0.05],
           [0.02, 0.95]])}
        """
        variable_CPD = {
            definition.find("FOR").text: list(map(float, table.text.split()))
            for definition in self.network.findall("DEFINITION")
            for table in definition.findall("TABLE")
        }
        for variable in variable_CPD:
            arr = np.array(variable_CPD[variable])
            arr = arr.reshape(
                (
                    len(self.variable_states[variable]),
                    arr.size // len(self.variable_states[variable]),
                ),
                order="F",
            )
            variable_CPD[variable] = arr
        return variable_CPD

    def get_property(self):
        """
        Returns the property of the variable

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_property() # doctest: +NORMALIZE_WHITESPACE
        {'asia': [None], 'bronc': [None], 'dysp': [None],
        'either': [None], 'lung': [None], 'smoke': [None],
        'tub': [None], 'xray': [None]}
        """
        variable_property = {
            variable.find("NAME").text: [property.text for property in variable.findall("PROPERTY")]
            for variable in self.network.findall("VARIABLE")
        }
        return variable_property

    def get_model(self, state_name_type=str):
        """
        Returns a Bayesian Network instance from the file/string.

        Parameters
        ----------
        state_name_type: int, str, or bool (default: str)
            The data type to which to convert the state names of the variables.

        Returns
        -------
        DiscreteBayesianNetwork instance: The read model.

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> model = reader.get_model()
        """
        model = DiscreteBayesianNetwork()
        model.add_nodes_from(self.variables)
        model.add_edges_from(self.edge_list)
        model.name = self.network_name

        tabular_cpds = []
        for var, values in self.variable_CPD.items():
            evidence_card = [len(self.variable_states[evidence_var]) for evidence_var in self.variable_parents[var]]
            cpd = TabularCPD(
                var,
                len(self.variable_states[var]),
                values,
                evidence=self.variable_parents[var],
                evidence_card=evidence_card,
                state_names={
                    var: list(map(state_name_type, self.state_names[var]))
                    for var in chain([var], self.variable_parents[var])
                },
            )
            tabular_cpds.append(cpd)

        model.add_cpds(*tabular_cpds)

        for node, properties in self.variable_property.items():
            for prop in properties:
                if prop is not None:
                    prop_name, prop_value = map(lambda t: t.strip(), prop.split("="))
                    model.nodes[node][prop_name] = prop_value

        return model


class XMLBIFWriter:
    """
    Initialise a XMLBIFWriter object.

    Parameters
    ----------
    model: DiscreteBayesianNetwork Instance
        Model to write

    encoding: str (optional)
        Encoding for text data

    prettyprint: Bool(optional)
        Indentation in output XML if true

    Examples
    --------
    >>> from pgmpy.readwrite import XMLBIFWriter
    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/asia")
    >>> writer = XMLBIFWriter(model)
    >>> writer.write("asia.xml")

    Reference
    ---------
    [1] https://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/
    """

    def __init__(self, model, encoding="utf-8", prettyprint=True):
        if not isinstance(model, DiscreteBayesianNetwork):
            raise TypeError("model must an instance of DiscreteBayesianNetwork")
        self.model = model

        self.encoding = encoding
        self.prettyprint = prettyprint

        self.xml = etree.Element("BIF", attrib={"VERSION": "0.3"})
        self.network = etree.SubElement(self.xml, "NETWORK")
        if self.model.name:
            etree.SubElement(self.network, "NAME").text = self.model.name
        else:
            etree.SubElement(self.network, "NAME").text = "UNTITLED"

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
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_variables() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'asia': <Element 'VARIABLE' at 0x...>,
        'bronc': <Element 'VARIABLE' at 0x...>,
        'dysp': <Element 'VARIABLE' at 0x...>,
        'either': <Element 'VARIABLE' at 0x...>,
        'lung': <Element 'VARIABLE' at 0x...>,
        'smoke': <Element 'VARIABLE' at 0x...>,
        'tub': <Element 'VARIABLE' at 0x...>,
        'xray': <Element 'VARIABLE' at 0x...>}
        """
        variables = self.model.nodes()
        variable_tag = {}
        for var in sorted(variables):
            variable_tag[var] = etree.SubElement(self.network, "VARIABLE", attrib={"TYPE": "nature"})
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
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_states() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'asia': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'bronc': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'dysp': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'either': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'lung': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'smoke': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'tub': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'xray': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>]}
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
                self.variable_name = var  # Set the current variable name
                state_tag.text = self._make_valid_state_name(state)
                outcome_tag[var].append(state_tag)
        return outcome_tag

    def _make_valid_state_name(self, state_name):
        """Transform the input state_name into a valid state in XMLBIF.
        XMLBIF states must start with a letter and only contain letters,
        numbers and underscores.
        """
        s = str(state_name)

        # Warn about commas in state names as they can cause issues when loading
        if "," in s:
            var_name = self.variable_name if hasattr(self, "variable_name") else "unknown"
            logger.warning(
                f"State name '{s}' for variable '{var_name}' contains commas. "
                "This may cause issues when loading the file. Consider removing any special characters."
            )

        # Keep existing transformation logic
        s_fixed = pp.CharsNotIn(pp.alphanums + "_").set_parse_action(pp.replace_with("_")).transform_string(s)
        if not s_fixed[0].isalpha():
            s_fixed = s_fixed

        if s != s_fixed:
            logger.warning(
                f"State name '{s}' has been modified to '{s_fixed}' to comply with XMLBIF format requirements. "
                "XMLBIF states must start with a letter and only contain letters, numbers, and underscores."  # noqa: E501
            )
        return s_fixed

    def get_properties(self):
        """
        Add property to variables in XMLBIF

        Return
        ------
        dict: dict of type {variable: property tag}

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_properties() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'asia': <Element 'PROPERTY' at 0x...>,
        'bronc': <Element 'PROPERTY' at 0x...>,
        'dysp': <Element 'PROPERTY' at 0x...>,
        'either': <Element 'PROPERTY' at 0x...>,
        'lung': <Element 'PROPERTY' at 0x...>,
        'smoke': <Element 'PROPERTY' at 0x...>,
        'tub': <Element 'PROPERTY' at 0x...>,
        'xray': <Element 'PROPERTY' at 0x...>}
        """
        variables = self.model.nodes()
        property_tag = {}
        for var in sorted(variables):
            properties = self.model.nodes[var]
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
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_definition() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'asia': <Element 'DEFINITION' at 0x...>,
        'bronc': <Element 'DEFINITION' at 0x...>,
        'dysp': <Element 'DEFINITION' at 0x...>,
        'either': <Element 'DEFINITION' at 0x...>,
        'lung': <Element 'DEFINITION' at 0x...>,
        'smoke': <Element 'DEFINITION' at 0x...>,
        'tub': <Element 'DEFINITION' at 0x...>,
        'xray': <Element 'DEFINITION' at 0x...>}
        """
        cpds = self.model.get_cpds()
        cpds.sort(key=lambda x: x.variable)
        definition_tag = {}
        for cpd in cpds:
            definition_tag[cpd.variable] = etree.SubElement(self.network, "DEFINITION")
            etree.SubElement(definition_tag[cpd.variable], "FOR").text = cpd.variable
            for parent in cpd.variables[1:]:
                etree.SubElement(definition_tag[cpd.variable], "GIVEN").text = parent

        return definition_tag

    def get_values(self):
        """
        Add Table to XMLBIF.

        Return
        ---------------
        dict: dict of type {variable: table tag}

        Examples
        -------
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_values() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'asia': <Element 'TABLE' at 0x...>,
        'bronc': <Element 'TABLE' at 0x...>,
        'dysp': <Element 'TABLE' at 0x...>,
        'either': <Element 'TABLE' at 0x...>,
        'lung': <Element 'TABLE' at 0x...>,
        'smoke': <Element 'TABLE' at 0x...>,
        'tub': <Element 'TABLE' at 0x...>,
        'xray': <Element 'TABLE' at 0x...>}
        """
        cpds = self.model.get_cpds()
        definition_tag = self.definition
        table_tag = {}
        for cpd in cpds:
            table_tag[cpd.variable] = etree.SubElement(definition_tag[cpd.variable], "TABLE")
            table_tag[cpd.variable].text = ""
            for val in compat_fns.ravel_f(cpd.get_values()):
                table_tag[cpd.variable].text += str(val) + " "

        return table_tag

    def write(self, filename):
        """
        Write the xml data into the file.

        Parameters
        ----------
        filename: Name of the file.

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("asia.xml")
        """
        with open(filename, "w") as fout:
            fout.write(self.__str__())

    def write_xmlbif(self, filename):
        warnings.warn(
            """`XMLBIFWriter.write_xmlbif` is deprecated and will be removed in v1.3.0. Please use `XMLBIFWriter.write`
            instead.""",
            FutureWarning,
            stacklevel=2,
        )
        self.write(filename)
