import re
from string import Template

import numpy
from pyparsing import Word, alphanums, Suppress, Optional, CharsNotIn, Group, nums, ZeroOrMore, OneOrMore,\
    cppStyleComment, printables

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.extern.six.moves import map, range


class BIFReader(object):

    """
    Base class for reading network file in bif format
    """

    def __init__(self, path=None, string=None):
        """
        Initialisation of BifReader object

        Parameters
        ----------------
        path : file or str
                File of bif data
        string : str
                String of bif data
        Examples
        -----------------
        # dog-problem.bif file is present at
        # http://www.cs.cmu.edu/~javabayes/Examples/DogProblem/dog-problem.bif
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader = BIFReader("bif_test.bif")
        <pgmpy.readwrite.BIF.BIFReader object at 0x7f2375621cf8>
        """
        if path:
            with open(path, 'r') as network:
                self.network = network.read()

        elif string:
            self.network = string

        else:
            raise ValueError("Must specify either path or string")

        if '"' in self.network:
            # Replacing quotes by spaces to remove case sensitivity like:
            # "Dog-Problem" and Dog-problem
            # or "true""false" and "true" "false" and true false
            self.network = self.network.replace('"', ' ')

        if '/*' in self.network or '//' in self.network:
            self.network = cppStyleComment.suppress().transformString(self.network)  # removing comments from the file

        self.name_expr, self.state_expr, self.property_expr = self.get_variable_grammar()
        self.probability_expr, self.cpd_expr = self.get_probability_grammar()
        self.network_name = self.get_network_name()
        self.variable_names = self.get_variables()
        self.variable_states = self.get_states()
        self.variable_properties = self.get_property()
        self.variable_parents = self.get_parents()
        self.variable_cpds = self.get_values()
        self.variable_edges = self.get_edges()

    def get_variable_grammar(self):
        """
         A method that returns variable grammar
        """
        # Defining a expression for valid word
        word_expr = Word(alphanums + '_' + '-')
        word_expr2 = Word(initChars=printables, excludeChars=['{', '}', ',', ' '])
        name_expr = Suppress('variable') + word_expr + Suppress('{')
        state_expr = ZeroOrMore(word_expr2 + Optional(Suppress(",")))
        # Defining a variable state expression
        variable_state_expr = Suppress('type') + Suppress(word_expr) + Suppress('[') + Suppress(Word(nums)) + \
            Suppress(']') + Suppress('{') + Group(state_expr) + Suppress('}') + Suppress(';')
        # variable states is of the form type description [args] { val1, val2 }; (comma may or may not be present)

        property_expr = Suppress('property') + CharsNotIn(';') + Suppress(';')  # Creating a expr to find property

        return name_expr, variable_state_expr, property_expr

    def get_probability_grammar(self):
        """
        A method that returns probability grammar
        """
        # Creating valid word expression for probability, it is of the format
        # wor1 | var2 , var3 or var1 var2 var3 or simply var
        word_expr = Word(alphanums + '-' + '_') + Suppress(Optional("|")) + Suppress(Optional(","))
        word_expr2 = Word(initChars=printables, excludeChars=[',', ')', ' ', '(']) + Suppress(Optional(","))
        # creating an expression for valid numbers, of the format
        # 1.00 or 1 or 1.00. 0.00 or 9.8e-5 etc
        num_expr = Word(nums + '-' + '+' + 'e' + 'E' + '.') + Suppress(Optional(","))
        probability_expr = Suppress('probability') + Suppress('(') + OneOrMore(word_expr) + Suppress(')')
        optional_expr = Suppress('(') + Suppress(OneOrMore(word_expr2)) + Suppress(')')
        probab_attributes = optional_expr | Suppress('table')
        cpd_expr = probab_attributes + OneOrMore(num_expr)

        return probability_expr, cpd_expr

    def variable_block(self):
        start = re.finditer('variable', self.network)
        for index in start:
            end = self.network.find('}\n', index.start())
            yield self.network[index.start():end]

    def probability_block(self):
        start = re.finditer('probability', self.network)
        for index in start:
            end = self.network.find('}\n', index.start())
            yield self.network[index.start():end]

    def get_network_name(self):
        """
        Retruns the name of the network

        Example
        ---------------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIF.BifReader("bif_test.bif")
        >>> reader.network_name()
        'Dog-Problem'
        """
        start = self.network.find('network')
        end = self.network.find('}\n', start)
        # Creating a network attribute
        network_attribute = Suppress('network') + Word(alphanums + '_' + '-') + '{'
        network_name = network_attribute.searchString(self.network[start:end])[0][0]

        return network_name

    def get_variables(self):
        """
        Returns list of variables of the network

        Example
        -------------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_variables()
        ['light-on','bowel_problem','dog-out','hear-bark','family-out']
        """
        variable_names = []
        for block in self.variable_block():
            name = self.name_expr.searchString(block)[0][0]
            variable_names.append(name)

        return variable_names

    def get_states(self):
        """
        Returns the states of variables present in the network

        Example
        -----------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_states()
        {'bowel-problem': ['true','false'],
        'dog-out': ['true','false'],
        'family-out': ['true','false'],
        'hear-bark': ['true','false'],
        'light-on': ['true','false']}
        """
        variable_states = {}
        for block in self.variable_block():
            name = self.name_expr.searchString(block)[0][0]
            variable_states[name] = list(self.state_expr.searchString(block)[0][0])

        return variable_states

    def get_property(self):
        """
        Returns the property of the variable

        Example
        -------------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_property()
        {'bowel-problem': ['position = (335, 99)'],
        'dog-out': ['position = (300, 195)'],
        'family-out': ['position = (257, 99)'],
        'hear-bark': ['position = (296, 268)'],
        'light-on': ['position = (218, 195)']}
        """
        variable_properties = {}
        for block in self.variable_block():
            name = self.name_expr.searchString(block)[0][0]
            properties = self.property_expr.searchString(block)
            variable_properties[name] = [y.strip() for x in properties for y in x]
        return variable_properties

    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Example
        --------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_parents()
        {'bowel-problem': [],
        'dog-out': ['family-out', 'bowel-problem'],
        'family-out': [],
        'hear-bark': ['dog-out'],
        'light-on': ['family-out']}
        """
        variable_parents = {}
        for block in self.probability_block():
            names = self.probability_expr.searchString(block.split('\n')[0])[0]
            variable_parents[names[0]] = names[1:]
        return variable_parents

    def get_values(self):
        """
        Returns the CPD of the variables present in the network

        Example
        --------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_values()
        {'bowel-problem': np.array([[0.01],
                                    [0.99]]),
        'dog-out': np.array([[0.99, 0.97, 0.9, 0.3],
                            [0.01, 0.03, 0.1, 0.7]]),
        'family-out': np.array([[0.15],
                                [0.85]]),
        'hear-bark': np.array([[0.7, 0.01],
                                [0.3, 0.99]]),
        'light-on': np.array([[0.6, 0.05],
                            [0.4, 0.95]])}
         """
        variable_cpds = {}
        for block in self.probability_block():
            name = self.probability_expr.searchString(block)[0][0]
            cpds = self.cpd_expr.searchString(block)
            arr = [float(j) for i in cpds for j in i]
            if 'table' in block:
                arr = numpy.array(arr)
                arr = arr.reshape((len(self.variable_states[name]),
                                   arr.size // len(self.variable_states[name])))
            else:
                length = len(self.variable_states[name])
                reshape_arr = [[] for i in range(length)]
                for i, val in enumerate(arr):
                    reshape_arr[i % length].append(val)
                arr = reshape_arr
                arr = numpy.array(arr)
            variable_cpds[name] = arr

        return variable_cpds

    def get_edges(self):
        """
        Returns the edges of the network

        Example
        --------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_edges()
        [['family-out', 'light-on'],
         ['family-out', 'dog-out'],
         ['bowel-problem', 'dog-out'],
         ['dog-out', 'hear-bark']]
        """
        edges = [[value, key] for key in self.variable_parents.keys()
                 for value in self.variable_parents[key]]
        return edges

    def get_model(self):
        """
        Returns the fitted bayesian model

        Example
        ----------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_model()
        <pgmpy.models.BayesianModel.BayesianModel object at 0x7f20af154320>
        """
        try:
            model = BayesianModel(self.variable_edges)
            model.name = self.network_name
            model.add_nodes_from(self.variable_names)

            tabular_cpds = []
            for var in sorted(self.variable_cpds.keys()):
                values = self.variable_cpds[var]
                cpd = TabularCPD(var, len(self.variable_states[var]), values,
                                 evidence=self.variable_parents[var],
                                 evidence_card=[len(self.variable_states[evidence_var])
                                                for evidence_var in self.variable_parents[var]])
                tabular_cpds.append(cpd)

            model.add_cpds(*tabular_cpds)
            for node, properties in self.variable_properties.items():
                for prop in properties:
                    prop_name, prop_value = map(lambda t: t.strip(), prop.split('='))
                    model.node[node][prop_name] = prop_value

            return model

        except AttributeError:
            raise AttributeError('First get states of variables, edges, parents and network name')


class BIFWriter(object):

    """
    Base class for writing BIF network file format
    """

    def __init__(self, model):
        """
        Initialise a BIFWriter Object

        Parameters
        ----------
        model: BayesianModel Instance

        Examples
        ---------
        >>> from pgmpy.readwrite import BIFWriter
        >>> writer = BIFWriter(model)
        >>> writer
        <writer_BIF.BIFWriter at 0x7f05e5ea27b8>
        """
        if not isinstance(model, BayesianModel):
            raise TypeError("model must be an instance of BayesianModel")
        self.model = model
        if not self.model.name:
            self.network_name = 'unknown'
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
        network_template = Template('network $name {\n}\n')
        # property tag may or may not be present in model,and since no of properties
        # can be more than one , will replace them accoriding to format otherwise null
        variable_template = Template("""variable $name {
    type discrete [ $no_of_states ] { $states };
$properties}\n""")
        property_template = Template('    property $prop ;\n')
        # $variable_ here is name of variable, used underscore for clarity
        probability_template = Template("""probability ( $variable_$seprator_$parents ) {
    table $values ;
}\n""")
        return network_template, variable_template, property_template, probability_template

    def __str__(self):
        """
        Returns the BIF format as string
        """
        network_template, variable_template, property_template, probability_template = self.BIF_templates()
        network = ''
        network += network_template.substitute(name=self.network_name)
        variables = self.model.nodes()

        for var in sorted(variables):
            no_of_states = str(len(self.variable_states[var]))
            states = ', '.join(self.variable_states[var])
            if not self.property_tag[var]:
                properties = ''
            else:
                properties = ''
                for prop_val in self.property_tag[var]:
                    properties += property_template.substitute(prop=prop_val)
            network += variable_template.substitute(name=var, no_of_states=no_of_states,
                                                    states=states, properties=properties)

        for var in sorted(variables):
            if not self.variable_parents[var]:
                parents = ''
                seprator = ''
            else:
                parents = ', '.join(self.variable_parents[var])
                seprator = ' | '
            cpd = ', '.join(map(str, self.tables[var]))
            network += probability_template.substitute(variable_=var, seprator_=seprator,
                                                       parents=parents, values=cpd)

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
        >>> model = BIFReader('dog-problem.bif').get_model()
        >>> writer = BIFWriter(model)
        >>> writer.get_variables()
        ['bowel-problem', 'family-out', 'hear-bark', 'light-on', 'dog-out']
        """
        variables = self.model.nodes()
        return variables

    def get_states(self):
        """
        Add states to variable of BIF

        Returns
        -------
        dict: dict of type {variable: a list of states}

        Example
        -------
        >>> from pgmpy.readwrite import BIFReader, BIFWriter
        >>> model = BIFReader('dog-problem.bif').get_model()
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
            for state in range(cpd.get_cardinality([variable])[variable]):
                variable_states[variable].append(str(variable) + '_' + str(state))
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
        >>> model = BIFReader('dog-problem.bif').get_model()
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
            properties = self.model.node[variable]
            property_tag[variable] = []
            for prop, val in properties.items():
                property_tag[variable].append(str(prop) + " = " + str(val))
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
        >>> model = BIFReader('dog-problem.bif').get_model()
        >>> writer = BIFWriter(model)
        >>> writer.get_parents()
        {'bowel-problem': [],
         'dog-out': ['bowel-problem', 'family-out'],
         'family-out': [],
         'hear-bark': ['dog-out'],
         'light-on': ['family-out']}
        """
        cpds = self.model.get_cpds()
        cpds.sort(key=lambda x: x.variable)
        variable_parents = {}
        for cpd in cpds:
            variable_parents[cpd.variable] = []
            for parent in sorted(cpd.variables[:0:-1]):
                variable_parents[cpd.variable].append(parent)
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
        >>> model = BIFReader('dog-problem.bif').get_model()
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
            tables[cpd.variable] = cpd.values.ravel()
        return tables

    def write_bif(self, filename):
        """
        Writes the BIF data into a file

        Parameters
        ----------
        filename : Name of the file

        Example
        -------
        >>> from pgmpy.readwrite import BIFReader, BIFWriter
        >>> model = BIFReader('dog-problem.bif').get_model()
        >>> writer = BIFWriter(model)
        >>> writer.write_bif(filname='test_file.bif')
        """
        writer = self.__str__()
        with open(filename, 'w') as fout:
            fout.write(writer)
