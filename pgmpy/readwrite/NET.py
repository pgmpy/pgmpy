import collections
from math import prod
from string import Template

import numpy as np
from pyparsing import (
    CharsNotIn,
    Group,
    OneOrMore,
    Optional,
    Suppress,
    Word,
    ZeroOrMore,
    alphanums,
    alphas,
    cppStyleComment,
    nums,
    printables,
)

from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.utils import compat_fns


class NETWriter(object):
    """
    Base class for writing network file in net format

    Parameters
    ----------
    model: BayesianNetwork Instance

    Examples
    ----------
    >>> from pgmpy.readwrite import NETWriter
    >>> from pgmpy.utils import get_example_model
    >>> asia = get_example_model('asia')
    >>> writer = NETWriter(asia)
    >>> writer
    <pgmpy.readwrite.NET.NETWriter at 0x7feac652c2b0>
    >>> writer.write_net('asia.net')

    Reference
    ---------
    [1] HUGIN EXPERT A/S . The HUGIN file format. http://www.hugin.com, 2011.
    """

    def __init__(self, model):
        if not isinstance(model, BayesianNetwork):
            raise TypeError("model must be an instance of BayesianNetwork")

        self.model = model

        if not self.model.name:
            self.network_name = "unknown"
        else:
            self.network_name = self.model.name

        self.variables = self.get_variables()
        self.variable_states = self.get_states()
        self.property_tag = self.get_properties()
        self.variable_parents = self.get_parents()
        self.tables = self.get_cpds()

    def NET_templates(self):
        """
        Create template for writing in NET format
        """

        network_template = Template("net {\n}\n")
        node_template = Template("node $name{\n    states = ($states);\n$properties}\n")
        potential_template = Template(
            "potential ($variable_$separator_$parents){\n data = $values;\n}\n"
        )
        property_template = Template("    $prop;\n")

        return (network_template, node_template, potential_template, property_template)

    def __str__(self):
        """Return the NET"""
        (
            network_template,
            node_template,
            potential_template,
            property_template,
        ) = self.NET_templates()

        network = ""
        network += network_template.substitute()
        variables = self.variables

        for var in sorted(variables):
            quoted_states = ['"' + state + '"' for state in self.variable_states[var]]
            states = "  ".join(quoted_states)

            if not self.property_tag[var]:
                properties = ""
            else:
                properties = ""
                for prop_val in self.property_tag[var]:
                    properties += property_template.substitute(prop=prop_val)

            network += node_template.substitute(
                name=var, states=states, properties=properties
            )

        for var in sorted(variables):
            if not self.variable_parents[var]:
                parents = ""
                separator = " |"
            else:
                parents = " ".join(self.variable_parents[var])
                separator = " | "
            potentials = self.net_cpd(var)
            network += potential_template.substitute(
                variable_=var,
                separator_=separator,
                parents=parents,
                values=potentials,
            )

        return network

    def net_cpd(self, var_name):
        """
        Util function for turning pgmpy CPT values into CPT format of .net files
        Inputs
        -------
        var_name: string, name of the variable

        Returns
        -------
        string: CPT format of .net files
        """
        cpt = self.tables[var_name]
        cpt_array = np.moveaxis(compat_fns.to_numpy(cpt, decimals=4), 0, -1)
        cpt_string = str(cpt_array)
        net_cpt_string = (
            cpt_string.replace("[", "(")
            .replace("]", ")")
            .replace(". ", ".0 ")
            .replace(".)", ".0)")
        )
        # Genie does not read potentials such as 1. therefore last line adds .0 to those
        return net_cpt_string

    def get_variables(self):
        """
        Add variables to NET

        Returns
        -------
        list: a list containing names of variable

        Example
        -------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = get_example_model('asia')
        >>> writer = NETWriter(asia)
        >>> writer.get_variables()
        ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
        """
        variables = list(self.model.nodes())
        return variables

    def get_cpds(self):
        """
        Adds tables to NET

        Returns
        -------
        dict: dict of type {variable: array}

        Example
        -------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = get_example_model('asia')
        >>> writer = NETWriter(asia)
        >>> writer.get_cpds()
        {'asia': array([0.01, 0.99]),
        'bronc': array([[0.6, 0.3],
                [0.4, 0.7]]),
        'dysp': array([[[0.9, 0.8],
                [0.7, 0.1]],

                [[0.1, 0.2],
                [0.3, 0.9]]]),
        'either': array([[[1., 1.],
                [1., 0.]],

                [[0., 0.],
                [0., 1.]]]),
        'lung': array([[0.1 , 0.01],
                [0.9 , 0.99]]),
        'smoke': array([0.5, 0.5]),
        'tub': array([[0.05, 0.01],
                [0.95, 0.99]]),
        'xray': array([[0.98, 0.05],
                [0.02, 0.95]])}
        """
        cpds = self.model.get_cpds()
        tables = {}
        for cpd in cpds:
            tables[cpd.variable] = cpd.values
        return tables

    def get_properties(self):
        """
        Add property to variables in NET

        Returns
        -------
        dict: dict of type {variable: list of properties }

        Example
        -------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = get_example_model('asia')
        >>> writer = NETWriter(asia)
        >>> writer.get_properties()
        """
        variables = self.model.nodes()
        property_tag = {}
        for variable in sorted(variables):
            properties = self.model.nodes[variable]
            properties = collections.OrderedDict(sorted(properties.items()))
            property_tag[variable] = []
            for prop, val in properties.items():
                property_tag[variable].append(str(prop) + " = " + str(val))
        return property_tag

    def get_states(self):
        """
        Add states to variable of NET

        Returns
        -------
        dict: dict of type {variable: a list of states}


        Example
        -------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = get_example_model('asia')
        >>> writer = NETWriter(asia)
        >>> writer.get_states()
        {'asia': ['yes', 'no'],
        'bronc': ['yes', 'no'],
        'dysp': ['yes', 'no'],
        'either': ['yes', 'no'],
        'lung': ['yes', 'no'],
        'smoke': ['yes', 'no'],
        'tub': ['yes', 'no'],
        'xray': ['yes', 'no']}
        """

        variable_states = {}
        cpds = self.model.get_cpds()
        for cpd in cpds:
            variable = cpd.variable
            variable_states[variable] = []
            for state in cpd.state_names[variable]:
                variable_states[variable].append(str(state))
        return variable_states

    def get_parents(self):
        """
        Add the parents to NET

        Returns
        -------
        dict: dict of type {variable: a list of parents}

        Example
        -------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = get_example_model('asia')
        >>> writer = NETWriter(asia)
        >>> writer.get_parents()
        {'asia': [],
        'bronc': ['smoke'],
        'dysp': ['bronc', 'either'],
        'either': ['lung', 'tub'],
        'lung': ['smoke'],
        'smoke': [],
        'tub': ['asia'],
        'xray': ['either']}
        """
        cpds = self.model.get_cpds()
        variable_parents = {}
        for cpd in cpds:
            variable_parents[cpd.variable] = cpd.variables[1:]
        return variable_parents

    def write_net(self, filename):
        """
        Writes the NET data into a file

        Parameters
        ----------
        filename : Name of the file

        Example
        -------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = get_example_model('asia')
        >>> writer = NETWriter(asia)
        >>> writer.write_net(filename='asia.net')
        """
        writer = self.__str__()
        with open(filename, "w") as fout:
            fout.write(writer)


class NETReader:
    """
    Initializes a NETReader object.

    Parameters
    ----------
    path : file or str
        File of net data

    string : str
        String of net data

    include_properties: boolean
        If True, gets the properties tag from the file and stores in graph properties.

    defaultname: int (default: "bn_model")
        Default name for the network if a network name is not available in the net file.

    Examples
    --------
    # asia.net file is present at
    # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
    >>> from pgmpy.readwrite import NETReader
    >>> reader = NETReader("asia.net")
    >>> reader
    <pgmpy.readwrite.NET.NETReader at 0x7feac645c640>
    >>> model = reader.get_model()
    """

    def __init__(
        self, path=None, string=None, include_properties=False, defaultName="bn_model"
    ):
        if path:
            with open(path, "r") as network:
                self.network = network.read()

        elif string:
            self.network = string

        else:
            raise ValueError("Must specify either path or string")

        self.include_properties = include_properties

        if "/*" in self.network or "//" in self.network:
            self.network = cppStyleComment.suppress().transformString(
                self.network
            )  # removing comments from the file

        (
            self.name_expr,
            self.state_expr,
            self.property_expr,
        ) = self.get_variable_grammar()

        self.potential_expr, self.cpd_expr = self.get_probability_grammar()

        if not self.get_network_name():
            self.network_name = defaultName
        else:
            self.network_name = self.get_network_name()

        self.variable_names = self.get_variables()
        self.variable_states = self.get_states()
        if self.include_properties:
            self.variable_properties = self.get_property()
        self.variable_parents = self.get_parents()
        self.variable_cpds = self.get_values()
        self.edges = self.get_edges()

    def get_variable_grammar(self):
        """
        A method that returns variable grammar
        """
        # Defining an expression for valid word
        word_expr = Word(alphanums + "_" + "-")("nodename")
        name_expr = Suppress("node ") + word_expr + Optional(Suppress("{"))

        word_expr2 = Word(initChars=printables, excludeChars=["(", ")", ",", " "])
        state_expr = ZeroOrMore(word_expr2 + Optional(Suppress(",")))
        # Defining a variable state expression
        variable_state_expr = (
            Suppress("states")
            + Suppress("=")
            + Suppress("(")
            + Group(state_expr)("statenames")
            + Suppress(")")
            + Suppress(";")
        )
        # variable states is of the form type description [args] { val1, val2 }; (comma may or may not be present)
        pexpr = Word(alphas.lower()) + Suppress("=") + CharsNotIn(";") + Suppress(";")
        property_expr = ZeroOrMore(pexpr)  # Creating an expr to find property

        variable_property_expr = (
            Suppress("node ")
            + Word(alphanums + "_" + "-")("varname")
            + Suppress("{")
            + Group(property_expr)("properties")
            + Suppress("}")
        )

        return name_expr, variable_state_expr, variable_property_expr

    def get_probability_grammar(self):
        """
        A method that returns probability grammar
        """

        word_expr = Word(alphanums + "-" + "_") + Suppress(Optional("|"))

        potential_expr = (
            Suppress("potential") + Suppress("(") + OneOrMore(word_expr) + Suppress(")")
        )

        num_expr = (
            Suppress(ZeroOrMore("("))
            + Word(nums + "-" + "+" + "e" + "E" + ".")
            + Suppress(ZeroOrMore(")"))
        )

        cpd_expr = Suppress("data") + Suppress("=") + OneOrMore(num_expr)

        return potential_expr, cpd_expr

    def get_network_name(self):
        """
        Returns the name of the network. Returns false if no network name is available

        Example
        ---------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> reader = NETReader("asia.net")
        >>> reader.get_network_name()
        False
        """

        start = self.network.find("net")
        end = self.network.find("}\n", start)
        # Creating a network attribute
        network_attribute = (
            Suppress("name")
            + Suppress("=")
            + Suppress('"')
            + Word(alphanums + "_" + "-")
            + Suppress('"')
            + Suppress(";")
        )
        network_name = network_attribute.searchString(self.network[start:end])
        if not network_name:
            return False
        return network_name[0][0]

    def get_variables(self):
        """
        Returns list of variables of the network

        Example
        ---------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> reader = NETReader("asia.net")
        >>> reader.get_variables()
        ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
        """
        variable_names = []

        for match in self.name_expr.scanString(self.network):
            result = match[0]
            name = result.nodename
            variable_names.append(name)

        return variable_names

    def get_states(self):
        """
        Returns the states of each variable in the network

        Example
        ---------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> reader = NETReader("asia.net")
        >>> reader.get_states()
        {'asia': ['yes', 'no'],
        'tub': ['yes', 'no'],
        'smoke': ['yes', 'no'],
        'lung': ['yes', 'no'],
        'bronc': ['yes', 'no'],
        'either': ['yes', 'no'],
        'xray': ['yes', 'no'],
        'dysp': ['yes', 'no']}
        """

        variable_states = {}
        for index, match in enumerate(self.name_expr.scanString(self.network)):
            result = match[0]
            name = result.nodename
            allstates = list(self.state_expr.scanString(self.network))
            states_unedited = list(
                allstates[index][0].statenames
            )  # includes double quotation like ['"state1"', '"state2"']
            states_edited = [state.replace('"', "") for state in states_unedited]
            variable_states[name] = states_edited
        return variable_states

    def get_property(self):
        """
        Returns the property of the variable

        Example
        -------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> reader = NETReader("asia.net")
        >>> reader.get_property()
        {'asia': {},
        'tub': {},
        'smoke': {},
        'lung': {},
        'bronc': {},
        'either': {},
        'xray': {},
        'dysp': {}}
        """

        variable_properties = {}
        for match in self.property_expr.scanString(self.network):
            var_name = match[0].varname
            prop_list = match[0].properties
            num_props = len(prop_list)
            props = {}
            for index in range(0, num_props, 2):
                props[prop_list[index].strip()] = prop_list[index + 1].strip()

            # Remove states from props
            props.pop("states", None)
            variable_properties[var_name] = props
        return variable_properties

    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Example
        -------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> reader = NETReader("asia.net")
        >>> reader.get_parents()
        {'asia': [],
        'tub': ['asia'],
        'smoke': [],
        'lung': ['smoke'],
        'bronc': ['smoke'],
        'either': ['lung', 'tub'],
        'xray': ['either'],
        'dysp': ['bronc', 'either']}
        """

        variable_parents = {}

        for match in self.potential_expr.scanString(self.network):
            vars_in_potential = match[0]
            variable_parents[vars_in_potential[0]] = vars_in_potential[1:]
        return variable_parents

    def get_values(self):
        """
        Returns the CPD of the variables present in the network

        Example
        -------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> reader = NETReader("asia.net")
        >>> reader.get_values()
        {'asia': array([[0.01],
                        [0.99]]),
        'tub': array([[0.05, 0.01],
                        [0.95, 0.99]]),
        'smoke': array([[0.5],
                        [0.5]]),
        'lung': array([[0.1 , 0.01],
                        [0.9 , 0.99]]),
        'bronc': array([[0.6, 0.3],
                        [0.4, 0.7]]),
        'either': array([[1., 1., 1., 0.],
                        [0., 0., 0., 1.]]),
        'xray': array([[0.98, 0.05],
                        [0.02, 0.95]]),
        'dysp': array([[0.9, 0.8, 0.7, 0.1],
                        [0.1, 0.2, 0.3, 0.9]])}
        """
        variable_cpds = {}

        parents = self.variable_parents
        variables = list(parents.keys())
        states = self.variable_states

        cpds = self.cpd_expr.scanString(self.network)

        for index, match in enumerate(cpds):
            var = variables[index]
            pars = parents[var]
            var_state_num = len(states[var])
            par_states_prod = prod([len(states[par]) for par in pars])
            cpd_flat = np.array(match[0], dtype="float64")
            cpd_2d = cpd_flat.reshape(par_states_prod, var_state_num).T

            variable_cpds[var] = cpd_2d

        return variable_cpds

    def get_edges(self):
        """
        Returns the edges of the network



        Example
        -------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> reader = NETReader("asia.net")
        >>> reader.get_edges()
        [['asia', 'tub'],
        ['smoke', 'lung'],
        ['smoke', 'bronc'],
        ['lung', 'either'],
        ['tub', 'either'],
        ['either', 'xray'],
        ['bronc', 'dysp'],
        ['either', 'dysp']]

        """
        edges = [
            [value, key]
            for key in self.variable_parents.keys()
            for value in self.variable_parents[key]
        ]
        return edges

    def get_model(self, state_name_type=str):
        """
        Returns the Bayesian Model read from the file/str.

        Parameters
        ----------
        state_name_type: int, str or bool (default: str)
            The data type to which to convert the state names of the variables.

        Example
        ----------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> reader = NETReader("asia.net")
        >>> reader.get_model()
        <pgmpy.models.BayesianNetwork.BayesianNetwork at 0x7febc059b430>
        """
        try:
            model = BayesianNetwork()
            model.add_nodes_from(self.variable_names)
            model.add_edges_from(self.edges)
            model.name = self.network_name

            tabular_cpds = []
            for var in sorted(self.variable_cpds.keys()):
                values = self.variable_cpds[var]
                states = self.variable_states[var]
                states_num = len(states)
                parents = self.variable_parents[var]
                parent_states_num = [len(self.variable_states[par]) for par in parents]

                state_names = {
                    par_var: list(map(state_name_type, self.variable_states[par_var]))
                    for par_var in parents
                }
                state_names[var] = list(map(state_name_type, states))

                cpd = TabularCPD(
                    var,
                    states_num,
                    values,
                    evidence=parents,
                    evidence_card=parent_states_num,
                    state_names=state_names,
                )
                tabular_cpds.append(cpd)

            model.add_cpds(*tabular_cpds)

            if self.include_properties:
                for node, properties in self.variable_properties.items():
                    for prop_name, prop_value in properties.items():
                        model.nodes[node][prop_name] = prop_value

            return model

        except AttributeError:
            raise AttributeError(
                "First get states of variables, edges, parents and network name"
            )
