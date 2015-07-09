from pyparsing import *
from itertools import combinations


class UAIReader:
    """
    Class for reading UAI file format from files or strings.
    """

    def __init__(self, path=None, string=None):
        """
        Initialize an instance of UAI reader class

        Parameters
        ----------
        path : file or str
            Path of the file containing UAI information.

        string : str
            String containing UAI information.

        Example
        -------
        >>> reader = UAIReader('TestUai.uai')

        Reference
        ---------
        http://graphmod.ics.uci.edu/uai08/FileFormat
        """
        if path:
            data = open(path)
            self.network = data.read()
        elif string:
            self.network = string
        else:
            raise ValueError("Must specify either path or string.")
        self.grammar = self.get_grammar()
        self.network_type = self.get_network_type()
        self.variables = self.get_variables()
        self.domain = self.get_domain()
        self.edges = self.get_edges()
        self.tables = self.get_tables()

    def get_grammar(self):
        """
        Returns the grammar of the UAI file.
        """
        network_name = Word(alphas).setResultsName('network_name')
        no_variables = Word(nums).setResultsName('no_variables')
        grammar = network_name+no_variables
        self.no_variables = int(grammar.parseString(self.network)['no_variables'])
        domain_variables = (Word(nums)*self.no_variables).setResultsName('domain_variables')
        grammar += domain_variables
        no_functions = Word(nums).setResultsName('no_functions')
        grammar += no_functions
        self.no_functions = int(grammar.parseString(self.network)['no_functions'])
        for function in range(0, self.no_functions):
            scope_grammar = Word(nums).setResultsName('fun_scope_'+str(function))
            grammar += scope_grammar
            function_scope = grammar.parseString(self.network)['fun_scope_'+str(function)]
            function_grammar = ((Word(nums))*int(function_scope)).setResultsName('fun_'+str(function))
            grammar += function_grammar

        floatnumber = Combine(Word(nums) + Optional(Literal(".") + Optional(Word(nums))))
        for function in range(0, self.no_functions):
            no_values_grammar = Word(nums).setResultsName('fun_no_values_' + str(function))
            grammar += no_values_grammar
            no_values = grammar.parseString(self.network)['fun_no_values_' + str(function)]
            values_grammar = ((floatnumber)*int(no_values)).setResultsName('fun_values_'+str(function))
            grammar += values_grammar
        return grammar

    def get_network_type(self):
        """
        Returns the type of network defined by the file.

        Returns
        -------
        string : str
            String containing network type.

        Example
        -------
        >>> reader = UAIReader('TestUAI.uai')
        >>> reader.get_network_type()
        'MARKOV'
        """
        network_type = self.grammar.parseString(self.network)
        return network_type['network_name']

    def get_variables(self):
        """
        Returns a list of variables.
        Each variable is represented by an index of list.
        For example if the no of variables are 4 then the list will be
        [var_0, var_1, var_2, var_3]

        Returns
        -------
        list: list of variables

        Example
        -------
        >>> reader = UAIReader('TestUAI.uai')
        >>> reader.get_variables()
        ['var_0', 'var_1', 'var_2']
        """
        variables = []
        for var in range(0, self.no_variables):
            var_name = "var_"+str(var)
            variables.append(var_name)
        return variables

    def get_domain(self):
        """
        Returns the dictionary of variables with keys as variable name
        and values as domain of the variables.

        Returns
        -------
        dict: dictionary containing variables and their domains

        Example
        -------
        >>> reader = UAIReader('TestUAI.uai')
        >>> reader.get_domain()
        {'var_0': '2', 'var_1': '2', 'var_2': '3'}
        """
        domain = {}
        var_domain = self.grammar.parseString(self.network)['domain_variables']
        for var in range(0, len(var_domain)):
            domain["var_"+str(var)] = var_domain[var]
        return domain

    def get_edges(self):
        """
        Returns the edges of the network.

        Returns
        -------
        set: set containing the edges of the network

        Example
        -------
        >>> reader = UAIReader('TestUAI.uai')
        >>> reader.get_edges()
        {('var_0', 'var_1'), ('var_0', 'var_2'), ('var_1', 'var_2')}
        """
        edges = []
        for function in range(0, self.no_functions):
            function_variables = self.grammar.parseString(self.network)['fun_'+str(function)]
            if self.network_type == 'BAYES':
                child_var = "var_" + str(function_variables[-1])
                function_variables = function_variables[:-1]
                for var in function_variables:
                    edges.append((child_var, "var_"+str(var)))
            elif self.network_type == "MARKOV":
                function_variables = ["var_" + var for var in function_variables]
                edges.extend(list(combinations(function_variables, 2)))
        return set(edges)

    def get_tables(self):
        """
        Returns list of tuple of child variable and CPD in case of Bayesian
        and list of tuple of scope of variables and values in case of Markov.

        Returns
        -------
        list : list of tuples of child variable and values in Bayesian
            list of tuples of scope of variables and values in case of Markov.

        Example
        -------
        >>> reader = UAIReader('TestUAI.uai')
        >>> reader.get_tables()
        [(['var_0', 'var_1'], ['4.000', '2.400', '1.000', '0.000']),
         (['var_0', 'var_1', 'var_2'],
          ['2.2500', '3.2500', '3.7500', '0.0000', '0.0000', '10.0000', '1.8750', '4.0000', '3.3330', '2.0000', '2.0000', '3.4000'])]
        """
        tables = []
        for function in range(0, self.no_functions):
            function_variables = self.grammar.parseString(self.network)['fun_'+str(function)]
            if self.network_type == 'BAYES':
                child_var = "var_" + str(function_variables[-1])
                values = self.grammar.parseString(self.network)['fun_values_'+str(function)]
                tables.append((child_var, list(values)))
            elif self.network_type == "MARKOV":
                function_variables = ["var_" + var for var in function_variables]
                values = self.grammar.parseString(self.network)['fun_values_'+str(function)]
                tables.append((function_variables, list(values)))
        return tables
