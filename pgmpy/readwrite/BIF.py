import re
import warnings
from itertools import product
from string import Template

import numpy as np
import pandas as pd
import pyparsing as pp

try:
    from pyparsing import (
        CharsNotIn,
        Group,
        OneOrMore,
        Optional,
        Suppress,
        Word,
        ZeroOrMore,
        nums,
    )
except ImportError as e:
    raise ImportError(
        f"{e}. pyparsing is required for using read/write methods. Please install using: pip install pyparsing."
    ) from None

from pgmpy import logger
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import compat_fns


class BIFReader:
    """
    Initializes a BIFReader object.

    Parameters
    ----------
    path : file or str
        File of bif data

    string : str
        String of bif data

    include_properties: boolean
        If True, gets the properties tag from the file and stores in graph properties.

    Examples
    --------
    >>> # dog-problem.bif file is present at
    >>> # http://www.cs.cmu.edu/~javabayes/Examples/DogProblem/dog-problem.bif
    >>> from pgmpy.readwrite import BIFReader
    >>> reader = BIFReader("bif_test.bif")
    <pgmpy.readwrite.BIF.BIFReader object at 0x7f2375621cf8>
    >>> model = reader.get_model()

    Reference
    ---------
    [1] Geoff Hulten and Pedro Domingos. The interchange format for bayesian networks.
        http://www.cs.washington.edu/dm/vfml/appendixes/bif.htm, 2003.
    """

    def __init__(self, path=None, string=None, include_properties=False):
        if path:
            with open(path) as network:
                self.network = network.read()

        elif string:
            self.network = string

        else:
            raise ValueError("Must specify either path or string")

        self.include_properties = include_properties

        if "/*" in self.network or "//" in self.network:
            # removing comments from the file
            pattern = r'("[^"\\]*(?:\\.[^"\\]*)*")|(/\*.*?\*/|//[^\n]*)'
            regex = re.compile(pattern, re.DOTALL)
            self.network = regex.sub(lambda m: m.group(1) if m.group(1) else "", self.network)

        if '"' in self.network:
            # Replacing quotes by spaces to remove case sensitivity like:
            # "Dog-Problem" and Dog-problem
            # or "true""false" and "true" "false" and true false
            self.network = self.network.replace('"', " ")

        (
            name_expr,
            state_expr,
            property_expr,
        ) = self.get_variable_grammar()
        probability_expr, cpd_expr = self.get_probability_grammar()

        # self.get_network_name()
        match = re.search(r"network\s+([\w-]+)\s*\{", self.network)
        self.network_name = match.group(1) if match else None

        block_pattern = re.compile(r"(variable|probability).*?\}\n", re.DOTALL)
        # Regex for parsing probability headers: handles spaces and dots in names
        prob_header_re = re.compile(r"probability\s*\(\s*(.+?)(?:\s*\|\s*(.+?))?\s*\)")
        # Regex for detecting table/default keywords (not inside state names)
        table_keyword_re = re.compile(r"(?:^|\{)\s*(table|default)\s", re.MULTILINE)
        # Regex for extracting state names from type declarations
        state_decl_re = re.compile(r"type\s+\w+\s*\[\s*\d+\s*\]\s*\{([^}]+)\}\s*;")

        self.variable_states = {}
        self.variable_names = []
        if self.include_properties:
            self.variable_properties = {}
        self.variable_parents = {}
        self.variable_edges = []
        probability_blocks = []

        for match in block_pattern.finditer(self.network):
            block_content = match.group(0)

            # self.get_variables(), self.get_states(), self.get_property()
            if block_content.startswith("variable"):
                name = name_expr.search_string(block_content)[0][0]
                self.variable_names.append(name)
                state_match = state_decl_re.search(block_content)
                raw_states = state_match.group(1)
                if "," in raw_states:
                    states = [s.strip() for s in raw_states.split(",") if s.strip()]
                else:
                    states = raw_states.split()
                self.variable_states[name] = states
                if self.include_properties:
                    properties = property_expr.search_string(block_content)
                    self.variable_properties[name] = [y.strip() for x in properties for y in x]

            # self.get_parents(), self.get_edges()
            elif block_content.startswith("probability"):
                header_line = block_content.split("\n")[0]
                prob_match = prob_header_re.search(header_line)
                var_name = prob_match.group(1).strip()
                if prob_match.group(2):
                    # Has | separator: supports multi-word names
                    parents = [p.strip() for p in prob_match.group(2).split(",")]
                else:
                    # No | separator: fall back to space-separated word splitting
                    # (old BIF format: first word is variable, rest are parents)
                    words = var_name.split()
                    if len(words) > 1:
                        var_name = words[0]
                        parents = list(words[1:])
                    else:
                        parents = []

                self.variable_parents[var_name] = parents
                self.variable_edges.extend([[p, var_name] for p in parents])
                probability_blocks.append((block_content, var_name, parents))

        # Normalize variable names in probability references to match declarations
        # (handles case mismatches like "neuroticism" vs "NEUROTICISM")
        name_map = {name.lower(): name for name in self.variable_names}
        normalized_parents = {}
        normalized_edges = []
        for var_name, parents in self.variable_parents.items():
            norm_var = name_map.get(var_name.lower(), var_name)
            norm_parents = [name_map.get(p.lower(), p) for p in parents]
            normalized_parents[norm_var] = norm_parents
            normalized_edges.extend([[p, norm_var] for p in norm_parents])
        self.variable_parents = normalized_parents
        self.variable_edges = normalized_edges
        probability_blocks = [
            (bc, name_map.get(vn.lower(), vn), [name_map.get(p.lower(), p) for p in ps])
            for bc, vn, ps in probability_blocks
        ]

        # self.get_values()
        self.variable_cpds = {}

        state_maps = {var: {state: i for i, state in enumerate(states)} for var, states in self.variable_states.items()}

        for block_content, var_name, parents in probability_blocks:
            cpds_list = cpd_expr.search_string(block_content)
            n_rows = len(self.variable_states[var_name])

            if table_keyword_re.search(block_content):
                arr = [float(j) for i in cpds_list for j in i]
                arr = np.array(arr).reshape(n_rows, -1)
                self.variable_cpds[var_name] = arr
            else:
                parent_cards = [len(self.variable_states[p]) for p in parents]
                arr_length = int(np.prod(parent_cards))
                arr = np.zeros((n_rows, arr_length))

                len_parents = len(parents)

                if len(cpds_list) > 0:
                    df = pd.DataFrame(cpds_list)

                    state_df = df.iloc[:, :len_parents].copy()
                    values_df = df.iloc[:, len_parents:]

                    for idx, parent in enumerate(parents):
                        col = state_df.columns[idx]
                        state_df = state_df.astype({col: "object"})
                        state_df.iloc[:, idx] = state_df.iloc[:, idx].map(state_maps[parent])

                    strides = np.cumprod([1] + parent_cards[::-1])[:-1][::-1]
                    col_indices = state_df.dot(strides).astype(int)
                    arr[:, col_indices] = values_df.astype(float).T
                self.variable_cpds[var_name] = arr

    def get_variable_grammar(self):
        """
        A method that returns variable grammar
        """
        # Variable name: everything between "variable" and "{", allowing spaces
        name_expr = Suppress("variable") + pp.Regex(r"[^{]+").set_parse_action(lambda t: t[0].strip()) + Suppress("{")
        # State names: comma-separated values that may contain spaces
        state_value = pp.Regex(r"[^,};]+").set_parse_action(lambda t: t[0].strip())
        # Defining a variable state expression
        variable_state_expr = (
            Suppress("type")
            + Suppress(Word(pp.unicode.alphanums + "_" + "-" + "."))
            + Suppress("[")
            + Suppress(Word(nums))
            + Suppress("]")
            + Suppress("{")
            + Group(state_value + ZeroOrMore(Suppress(",") + state_value))
            + Suppress("}")
            + Suppress(";")
        )
        # variable states is of the form type description [args] { val1, val2 }; (comma may or may not be present)

        property_expr = Suppress("property") + CharsNotIn(";") + Suppress(";")  # Creating an expr to find property

        return name_expr, variable_state_expr, property_expr

    def get_probability_grammar(self):
        """
        A method that returns probability grammar
        """
        # Creating valid word expression for probability, it is of the format
        # wor1 | var2 , var3 or var1 var2 var3 or simply var
        word_expr = Word(pp.unicode.alphanums + "-" + "_" + ".") + Suppress(Optional("|")) + Suppress(Optional(","))
        # creating an expression for valid numbers, of the format
        # 1.00 or 1 or 1.00. 0.00 or 9.8e-5 etc
        num_expr = Word(nums + "-" + "+" + "e" + "E" + ".") + Suppress(Optional(","))
        probability_expr = Suppress("probability") + Suppress("(") + OneOrMore(word_expr) + Suppress(")")
        # State values in CPD rows: comma-separated values that may contain spaces
        state_value = pp.Regex(r"[^,)]+").set_parse_action(lambda t: t[0].strip())
        optional_expr = Suppress("(") + state_value + ZeroOrMore(Suppress(",") + state_value) + Suppress(")")
        probab_attributes = optional_expr | Suppress("table") | Suppress("default")
        cpd_expr = probab_attributes + OneOrMore(num_expr)

        return probability_expr, cpd_expr

    def get_model(self, state_name_type=str):
        """
        Returns the Bayesian Model read from the file/str.

        Parameters
        ----------
        state_name_type: int, str or bool (default: str)
            The data type to which to convert the state names of the variables.

        Example
        ----------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_model()
        <pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork object at 0x7f20af154320>
        """
        model = DiscreteBayesianNetwork()
        model.add_nodes_from(self.variable_names)
        model.add_edges_from(self.variable_edges)
        model.name = self.network_name

        tabular_cpds = []
        for var in sorted(self.variable_cpds.keys()):
            values = self.variable_cpds[var]
            sn = {
                p_var: list(map(state_name_type, self.variable_states[p_var])) for p_var in self.variable_parents[var]
            }
            sn[var] = list(map(state_name_type, self.variable_states[var]))
            cpd = TabularCPD(
                var,
                len(self.variable_states[var]),
                values,
                evidence=self.variable_parents[var],
                evidence_card=[len(self.variable_states[evidence_var]) for evidence_var in self.variable_parents[var]],
                state_names=sn,
            )
            tabular_cpds.append(cpd)

        model.add_cpds(*tabular_cpds)

        if self.include_properties:
            for node, properties in self.variable_properties.items():
                for prop in properties:
                    prop_name, prop_value = map(lambda t: t.strip(), prop.split("="))
                    model.nodes[node][prop_name] = prop_value

        return model


class BIFWriter:
    """
    Initialise a BIFWriter Object

    Parameters
    ----------
    model: DiscreteBayesianNetwork Instance

    round_values: int (default: None)
        Round the probability values to `round_values` decimals. If None, keeps all decimal points.

    Examples
    ---------
    >>> from pgmpy.readwrite import BIFWriter
    >>> from pgmpy.example_models import load_model
    >>> asia = load_model("bnlearn/asia")
    >>> writer = BIFWriter(asia)
    >>> writer
    <writer_BIF.BIFWriter at 0x7f05e5ea27b8>
    >>> writer.write("asia.bif")
    """

    def __init__(self, model, round_values=None):
        if not isinstance(model, DiscreteBayesianNetwork):
            raise TypeError("model must be an instance of DiscreteBayesianNetwork")
        self.model = model
        self.round_values = round_values
        if not self.model.name:
            self.network_name = "unknown"
        else:
            self.network_name = self.model.name
        self.variable_states = self.get_states()
        self.property_tag = self.get_properties()
        self.variable_parents = self.get_parents()
        self.tables = self.get_cpds()

    def BIF_templates(self):
        """
        Create template for writing in BIF format
        """
        network_template = Template("network $name {\n}\n")
        # property tag may or may not be present in model,and since no of properties
        # can be more than one, will replace them according to format otherwise null
        variable_template = Template(
            """variable $name {
    type discrete [ $no_of_states ] { $states };
$properties}\n"""
        )
        property_template = Template("    property $prop ;\n")
        # $variable_ here is name of variable, used underscore for clarity
        probability_template = Template(
            """probability ( $variable_$separator_$parents ) {
    table $values ;
}\n"""
        )

        conditional_probability_template_total = Template(
            """probability ( $variable_$separator_$parents ) {
$values
}\n"""
        )

        conditional_probability_template = Template("""    ( $state ) $values;\n""")

        return (
            network_template,
            variable_template,
            property_template,
            probability_template,
            conditional_probability_template_total,
            conditional_probability_template,
        )

    def __str__(self):
        """
        Returns the BIF format as string
        """
        (
            network_template,
            variable_template,
            property_template,
            probability_template,
            conditional_probability_template_total,
            conditional_probability_template,
        ) = self.BIF_templates()
        network = ""
        network += network_template.substitute(name=self.network_name)
        variables = self.model.nodes()

        sorted_variables = sorted(variables)

        for var in sorted_variables:
            no_of_states = str(len(self.variable_states[var]))
            states = ", ".join(self.variable_states[var])
            if not self.property_tag[var]:
                properties = ""
            else:
                properties = ""
                for prop_val in self.property_tag[var]:
                    properties += property_template.substitute(prop=prop_val)
            network += variable_template.substitute(
                name=var,
                no_of_states=no_of_states,
                states=states,
                properties=properties,
            )

        for var in sorted_variables:
            if not self.variable_parents[var]:
                parents = ""
                separator = ""
                cpd = ", ".join(map(str, self.tables[var]))
                network += probability_template.substitute(
                    variable_=var, separator_=separator, parents=parents, values=cpd
                )
            else:
                parents_str = ", ".join(self.variable_parents[var])
                separator = " | "
                cpd = self.model.get_cpds(var)
                cpd_values_transpose = cpd.get_values().T

                # Get the sanitized state names for parents from self.variable_states
                parent_states = product(*[self.variable_states[var] for var in cpd.variables[1:]])
                all_cpd = ""
                for index, state in enumerate(parent_states):
                    all_cpd += conditional_probability_template.substitute(
                        state=", ".join(map(str, state)),
                        values=", ".join(
                            map(
                                str,
                                compat_fns.to_numpy(
                                    cpd_values_transpose[index, :],
                                    decimals=self.round_values,
                                ),
                            )
                        ),
                    )
                network += conditional_probability_template_total.substitute(
                    variable_=var,
                    separator_=separator,
                    parents=parents_str,
                    values=all_cpd,
                )
        return network

    def get_variables(self):
        """
        Add variables to BIF

        Returns
        -------
        list: a list containing names of variable

        Example
        -------
        >>> from pgmpy.readwrite import BIFReader, BIFWriter
        >>> model = BIFReader("dog-problem.bif").get_model()
        >>> writer = BIFWriter(model)
        >>> writer.get_variables()
        ['bowel-problem', 'family-out', 'hear-bark', 'light-on', 'dog-out']
        """
        variables = self.model.nodes()
        return variables

    def get_states(self):
        """
        Add states to variable of BIF, handling commas in state names by replacing them with underscores.

        Returns
        -------
        dict: dict of type {variable: a list of states}

        Example
        -------
        >>> from pgmpy.readwrite import BIFReader, BIFWriter
        >>> model = BIFReader("dog-problem.bif").get_model()
        >>> writer = BIFWriter(model)
        >>> writer.get_states()
        {'bowel-problem': ['bowel-problem_0', 'bowel-problem_1'],
         'dog-out': ['dog-out_0', 'dog-out_1'],
         'family-out': ['family-out_0', 'family-out_1'],
         'hear-bark': ['hear-bark_0', 'hear-bark_1'],
         'light-on': ['light-on_0', 'light-on_1']}
        """
        variable_states = {}
        cpds = self.model.get_cpds()
        for cpd in cpds:
            variable = cpd.variable
            variable_states[variable] = []
            for state in cpd.state_names[variable]:
                state_str = str(state)

                # Warn users if any commas in state names
                if "," in state_str:
                    logger.warning(
                        f"State name '{state_str}' for variable '{variable}' contains commas. "
                        "This may cause issues when loading the file. Consider removing any special characters."
                    )
                variable_states[variable].append(state_str)
        return variable_states

    def get_properties(self):
        """
        Add property to variables in BIF

        Returns
        -------
        dict: dict of type {variable: list of properties }

        Example
        -------
        >>> from pgmpy.readwrite import BIFReader, BIFWriter
        >>> model = BIFReader("dog-problem.bif").get_model()
        >>> writer = BIFWriter(model)
        >>> writer.get_properties()
        {'bowel-problem': ['position = (335, 99)'],
         'dog-out': ['position = (300, 195)'],
         'family-out': ['position = (257, 99)'],
         'hear-bark': ['position = (296, 268)'],
         'light-on': ['position = (218, 195)']}
        """
        variables = self.model.nodes()
        property_tag = {}
        for variable in sorted(variables):
            properties = self.model.nodes[variable]
            property_tag[variable] = [f"{prop} = {val}" for prop, val in sorted(properties.items())]
        return property_tag

    def get_parents(self):
        """
        Add the parents to BIF

        Returns
        -------
        dict: dict of type {variable: a list of parents}

        Example
        -------
        >>> from pgmpy.readwrite import BIFReader, BIFWriter
        >>> model = BIFReader("dog-problem.bif").get_model()
        >>> writer = BIFWriter(model)
        >>> writer.get_parents()
        {'bowel-problem': [],
         'dog-out': ['bowel-problem', 'family-out'],
         'family-out': [],
         'hear-bark': ['dog-out'],
         'light-on': ['family-out']}
        """
        cpds = self.model.get_cpds()
        variable_parents = {}
        for cpd in cpds:
            variable_parents[cpd.variable] = cpd.variables[1:]
        return variable_parents

    def get_cpds(self):
        """
        Adds tables to BIF

        Returns
        -------
        dict: dict of type {variable: array}

        Example
        -------
        >>> from pgmpy.readwrite import BIFReader, BIFWriter
        >>> model = BIFReader("dog-problem.bif").get_model()
        >>> writer = BIFWriter(model)
        >>> writer.get_cpds()
        {'bowel-problem': array([ 0.01,  0.99]),
         'dog-out': array([ 0.99,  0.97,  0.9 ,  0.3 ,  0.01,  0.03,  0.1 ,  0.7 ]),
         'family-out': array([ 0.15,  0.85]),
         'hear-bark': array([ 0.7 ,  0.01,  0.3 ,  0.99]),
         'light-on': array([ 0.6 ,  0.05,  0.4 ,  0.95])}
        """
        cpds = self.model.get_cpds()
        tables = {}
        for cpd in cpds:
            tables[cpd.variable] = compat_fns.to_numpy(cpd.values.ravel(), decimals=self.round_values)
        return tables

    def write(self, filename):
        """
        Writes the BIF data into a file

        Parameters
        ----------
        filename : Name of the file

        Example
        -------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.readwrite import BIFReader, BIFWriter
        >>> asia = load_model("bnlearn/asia")
        >>> writer = BIFWriter(asia)
        >>> writer.write(filename="asia.bif")
        """
        writer = self.__str__()
        with open(filename, "w") as fout:
            fout.write(writer)

    def write_bif(self, filename):
        warnings.warn(
            "`BIFWriter.write_bif` is deprecated. Please use `BIFWriter.write` instead.", FutureWarning, stacklevel=2
        )
        self.write(filename)
