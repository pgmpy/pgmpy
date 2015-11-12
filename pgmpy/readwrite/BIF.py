import numpy
from pyparsing import Word, alphanums, Suppress, Optional, CharsNotIn, Group, nums, ZeroOrMore, OneOrMore, cppStyleComment, Literal
import re
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD


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
        multi_space = OneOrMore(Literal(' ')|'\t'|'\r'|'\f')    # An pyparsing expression for checking mulitple spaces

        if path:
            with open(path, 'r') as network:
                self.network = network.read()

        elif string:
            self.network = string

        else:
            raise ValueError("Must specify either path or string")

        if '"' in self.network:
            """
            Replacing quotes by spaces to remove case sensitivity like:
            "Dog-Problem" and Dog-problem
            or "true""false" and "true" "false" and true false
            """
            self.network = self.network.replace('"', ' ')

        self.network = multi_space.setParseAction(lambda t:' ').transformString(self.network)   # replacing mulitple spaces or tabs by one space

        if '/*' in self.network or '//' in self.network:
            self.network = cppStyleComment.suppress().transformString(self.network)             # removing comments from the file

        self.grammar = self.get_grammar(self.network)
        self.network_name = self.get_network_name()
        self.variable_names = self.get_variables()
        self.variable_states = self.get_states()
        self.variable_properties = self.get_property()
        self.variable_cpds = self.get_cpd()
        self.variable_parents = self.get_parents()
        self.variable_edges = self.get_edges() 
        self.get_model()
 
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
        network_attribute = Suppress('network') + Word(alphanums+'_'+'-') + '{'   # Creating a network attribute 
        network_name = network_attribute.searchString(self.network)[0][0]
        return network_name


    class get_grammar(object):
    
        """
        A class to that finds grammar of the network
        """
        def __init__(self, network=None):

            self.net = network
            self.variable_names, self.variable_states, self.variable_properties = self.get_variable_grammar()
            self.variable_parents, self.variable_cpds = self.get_probability_grammar()

        def get_variable_grammar(self):

            """
            A method that returns variable grammar
            """
            variable_block_starts = [x.start() for x in re.finditer('variable', self.net)]
            variable_block = []

            for i in variable_block_starts:
                variable_block_end = self.net.find('}\n', i)
                variable_block.append(self.net[i:variable_block_end])

            variable_names = []
            variable_states = {}
            variable_properties = {}
            # Defining a expression for valid word
            word_expr = Word(alphanums+'_'+'-')
            name_expr = Suppress('variable') + word_expr + Suppress('{')
            state_expr = ZeroOrMore(word_expr + Optional(Suppress(",")))
            # Defining a variable state expression
            variable_state_expr = Suppress('type') + Suppress(word_expr) + Suppress('[') + Suppress(Word(nums)) + \
            Suppress(']') + Suppress('{') + Group(state_expr) + Suppress('}') + Suppress(';')
            # variable states is of the form type description [args] { val1, val2 }; (comma may or may not be present)

            property_expr = Suppress('property') + CharsNotIn(';') + Suppress(';')  # Creating a expr to find property

            for block in variable_block:
                name = name_expr.searchString(block)[0][0]
                variable_names.append(name) 
                variable_states[name] = list(variable_state_expr.searchString(block)[0][0])
                properties = property_expr.searchString(block)
                variable_properties[name] = [x[0].strip() for x in properties]

            return variable_names, variable_states, variable_properties
            
        def get_probability_grammar(self):
        
            """
            A method that returns probability grammar
            """
            probability_block_starts = [x.start() for x in re.finditer('probability', self.net)]
            probability_block = []
            for i in probability_block_starts:
                probability_block_end = self.net.find('}\n',i)
                probability_block.append(self.net[i:probability_block_end])

            # Creating valid word expression for probability, it is of the format
            # wor1 | var2 , var3 or var1 var2 var3 or simply var 
            word_expr = Word(alphanums + '-' + '_') + Suppress(Optional("|")) + Suppress(Optional(","))
            # creating an expression for valid numbers, of the format
            # 1.00 or 1 or 1.00. 0.00 or 9.8e-5 etc
            num_expr = Word(nums + '-' + '+' + 'e' +'E' +'.') + Suppress(Optional(","))
            probability_expr = Suppress('probability') + Suppress('(') + OneOrMore(word_expr) + Suppress(')')
            optional_expr = Suppress('(') + Suppress(OneOrMore(word_expr)) + Suppress(')')
            probab_attributes = optional_expr | Suppress('table')
            cpd_expr = probab_attributes + OneOrMore(num_expr)

            variable_parents = {}
            variable_cpds ={}

            for block in probability_block:
                names = probability_expr.searchString(block)
                names = names[0]
                variable = names[0]
                names = names[1:]
                variable_parents[variable] = names
                cpds = cpd_expr.searchString(block)
                arr = [float(j) for i in cpds for j in i]
                arr = numpy.array(arr)
                arr = arr.reshape((len(self.variable_states[variable]), 
                                  arr.size//len(self.variable_states[variable])))
                variable_cpds[variable] = arr

            return variable_parents, variable_cpds

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
        try:
            self.grammar.variable_names
        except AttributeError:
            self.grammar = self.get_grammar(self.network)
        return self.grammar.variable_names
               

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

        try:
            self.grammar.variable_states
        except AttributeError:
            self.grammar = self.get_grammar(self.network)
        return self.grammar.variable_states

    def get_property(self):

        """
        Returns the property of the variable

        Example
        -------------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_property()
        {{'bowel-problem': ['position = (335, 99)'],
          'dog-out': ['position = (300, 195)'],
          'family-out': ['position = (257, 99)'],
          'hear-bark': ['position = (296, 268)'],
          'light-on': ['position = (218, 195)']}
        """

        try:
            self.grammar.variable_properties
        except AttributeError:
            self.grammar = self.get_grammar(self.network)
        return self.grammar.variable_properties

    def get_cpd(self):

        """
        Returns the CPD of the variables present in the network

        Example
        --------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_cpd()
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
        try:
            self.grammar.variable_cpds
        except AttributeError:
            self.grammar = self.get_grammar(self.network)
        return self.grammar.variable_cpds

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
        try:
            self.grammar.variable_parents
        except AttributeError:
            self.grammar = self.get_grammar(self.network)
        return self.grammar.variable_parents

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
        try:
            self.grammar.variable_parents
        except AttributeError:
            self.grammar = self.get_grammar(self.network)
        edges = [[value, key] for key in self.grammar.variable_parents.keys()
                     for value in self.grammar.variable_parents[key]]
        return edges

    def get_model(self):

        """
        Returns the fitted bayesian model
        
        Example
        ----------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_model()
        'Dog-Problem'
        """
        try:
            model = BayesianModel(self.variable_edges)
            model.name = self.network_name

            tabular_cpds = []
            count_dict={}
            for var1,var2 in self.variable_edges:
                if count_dict.get(var1,0) == 0:
                    values = self.variable_cpds[var1]
                    cpd = TabularCPD(var1, len(self.variable_states[var1]), values,
                                    evidence = self.variable_parents[var1],
                                    evidence_card = [len(self.variable_states[evidence_var])
                                                    for evidence_var in self.variable_parents[var1]])
                    count_dict[var1] = 1
                    tabular_cpds.append(cpd)
                if count_dict.get(var2,0) == 0:
                    values = self.variable_cpds[var2]
                    cpd = TabularCPD(var2, len(self.variable_states[var2]), values,
                                    evidence = self.variable_parents[var2],
                                    evidence_card = [len(self.variable_states[evidence_var])
                                                    for evidence_var in self.variable_parents[var2]])
                    count_dict[var2] = 1
                    tabular_cpds.append(cpd)

            model.add_cpds(*tabular_cpds)

            for node, properties in self.variable_properties.items():
                for prop in properties:
                    prop_name, prop_value = map(lambda t: t.strip(), prop.split('='))
                    model.node[node][prop_name] = prop_value

            return model
        except AttributeError:
            raise AttributeError('First get states of variables, edges, parents and network name')
