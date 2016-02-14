from itertools import combinations

import numpy as np

from pyparsing import alphas, Combine, Literal, Optional, nums, Word

from pgmpy.models import BayesianModel, MarkovModel
from pgmpy.factors import TabularCPD, Factor
from pgmpy.extern.six.moves import map, range


class UAIReader(object):
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
        grammar = network_name + no_variables
        self.no_variables = int(grammar.parseString(self.network)['no_variables'])
        domain_variables = (Word(nums)*self.no_variables).setResultsName('domain_variables')
        grammar += domain_variables
        no_functions = Word(nums).setResultsName('no_functions')
        grammar += no_functions
        self.no_functions = int(grammar.parseString(self.network)['no_functions'])
        integer = Word(nums).setParseAction(lambda t: int(t[0]))
        for function in range(0, self.no_functions):
            scope_grammar = Word(nums).setResultsName('fun_scope_' + str(function))
            grammar += scope_grammar
            function_scope = grammar.parseString(self.network)['fun_scope_' + str(function)]
            function_grammar = ((integer)*int(function_scope)).setResultsName('fun_' + str(function))
            grammar += function_grammar

        floatnumber = Combine(Word(nums) + Optional(Literal(".") + Optional(Word(nums))))
        for function in range(0, self.no_functions):
            no_values_grammar = Word(nums).setResultsName('fun_no_values_' + str(function))
            grammar += no_values_grammar
            no_values = grammar.parseString(self.network)['fun_no_values_' + str(function)]
            values_grammar = ((floatnumber)*int(no_values)).setResultsName('fun_values_' + str(function))
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
            var_name = "var_" + str(var)
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
            domain["var_" + str(var)] = var_domain[var]
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
            function_variables = self.grammar.parseString(self.network)['fun_' + str(function)]
            if isinstance(function_variables, int):
                function_variables = [function_variables]
            if self.network_type == 'BAYES':
                child_var = "var_" + str(function_variables[-1])
                function_variables = function_variables[:-1]
                for var in function_variables:
                    edges.append((child_var, "var_" + str(var)))
            elif self.network_type == "MARKOV":
                function_variables = ["var_" + str(var) for var in function_variables]
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
          ['2.2500', '3.2500', '3.7500', '0.0000', '0.0000', '10.0000',
           '1.8750', '4.0000', '3.3330', '2.0000', '2.0000', '3.4000'])]
        """
        tables = []
        for function in range(0, self.no_functions):
            function_variables = self.grammar.parseString(self.network)['fun_' + str(function)]
            if isinstance(function_variables, int):
                function_variables = [function_variables]
            if self.network_type == 'BAYES':
                child_var = "var_" + str(function_variables[-1])
                values = self.grammar.parseString(self.network)['fun_values_' + str(function)]
                tables.append((child_var, list(values)))
            elif self.network_type == "MARKOV":
                function_variables = ["var_" + str(var) for var in function_variables]
                values = self.grammar.parseString(self.network)['fun_values_' + str(function)]
                tables.append((function_variables, list(values)))
        return tables

    def get_model(self):
        """
        Returns an instance of Bayesian Model or Markov Model.
        Varibles are in the pattern var_0, var_1, var_2 where var_0 is
        0th index variable, var_1 is 1st index variable.

        Return
        ------
        model: an instance of Bayesian or Markov Model.

        Examples
        --------
        >>> reader = UAIReader('TestUAI.uai')
        >>> reader.get_model()
        """
        if self.network_type == 'BAYES':
            model = BayesianModel(self.edges)

            tabular_cpds = []
            for cpd in self.tables:
                child_var = cpd[0]
                states = int(self.domain[child_var])
                arr = list(map(float, cpd[1]))
                values = np.array(arr)
                values = values.reshape(states, values.size // states)
                tabular_cpds.append(TabularCPD(child_var, states, values))

            model.add_cpds(*tabular_cpds)
            return model

        elif self.network_type == 'MARKOV':
            model = MarkovModel(self.edges)

            factors = []
            for table in self.tables:
                variables = table[0]
                cardinality = [int(self.domain[var]) for var in variables]
                value = list(map(float, table[1]))
                factor = Factor(variables=variables, cardinality=cardinality, values=value)
                factors.append(factor)

            model.add_factors(*factors)
            return model


class UAIWriter(object):
    """
    Class for writing models in UAI.
    """
    def __init__(self, model):
        """
        Initialize an instance of UAI writer class

        Parameters
        ----------
        model: A Bayesian or Markov model
            The model to write
        """
        if isinstance(model, BayesianModel):
            self.network = "BAYES\n"
        elif isinstance(model, MarkovModel):
            self.network = "MARKOV\n"
        else:
            raise TypeError("Model must be an instance of Bayesian or Markov model.")

        self.model = model
        self.no_nodes = self.get_nodes()
        self.domain = self.get_domain()
        self.functions = self.get_functions()
        self.tables = self.get_tables()

    def __str__(self):
        """
        Returns the UAI file as a string.
        """
        self.network += self.no_nodes + "\n"
        domain = sorted(self.domain.items(), key=lambda x: (x[1], x[0]))
        self.network += " ".join([var[1] for var in domain]) + "\n"
        self.network += str(len(self.functions)) + "\n"
        for fun in self.functions:
            self.network += str(len(fun)) + " "
            self.network += " ".join(fun) + "\n"
        self.network += "\n"
        for table in self.tables:
            self.network += str(len(table)) + "\n"
            self.network += " ".join(table) + "\n"
        return self.network[:-1]

    def get_nodes(self):
        """
        Adds variables to the network.

        Example
        -------
        >>> writer = UAIWriter(model)
        >>> writer.get_nodes()
        """
        no_nodes = len(self.model.nodes())
        return str(no_nodes)

    def get_domain(self):
        """
        Adds domain of each variable to the network.

        Example
        -------
        >>> writer = UAIWriter(model)
        >>> writer.get_domain()
        """
        if isinstance(self.model, BayesianModel):
            cpds = self.model.get_cpds()
            cpds.sort(key=lambda x: x.variable)
            domain = {}
            for cpd in cpds:
                domain[cpd.variable] = str(cpd.variable_card)
            return domain
        elif isinstance(self.model, MarkovModel):
            factors = self.model.get_factors()
            domain = {}
            for factor in factors:
                variables = factor.variables
                for var in variables:
                    if var not in domain:
                        domain[var] = str(factor.get_cardinality([var])[var])
            return domain
        else:
            raise TypeError("Model must be an instance of Markov or Bayesian model.")

    def get_functions(self):
        """
        Adds functions to the network.

        Example
        -------
        >>> writer = UAIWriter(model)
        >>> writer.get_functions()
        """
        if isinstance(self.model, BayesianModel):
            cpds = self.model.get_cpds()
            cpds.sort(key=lambda x: x.variable)
            variables = sorted(self.domain.items(), key=lambda x: (x[1], x[0]))
            functions = []
            for cpd in cpds:
                child_var = cpd.variable
                evidence = cpd.variables[:0:-1]
                function = [str(variables.index((var, self.domain[var]))) for var in evidence]
                function.append(str(variables.index((child_var, self.domain[child_var]))))
                functions.append(function)
            return functions
        elif isinstance(self.model, MarkovModel):
            factors = self.model.get_factors()
            functions = []
            variables = sorted(self.domain.items(), key=lambda x: (x[1], x[0]))
            for factor in factors:
                scope = factor.scope()
                function = [str(variables.index((var, self.domain[var]))) for var in scope]
                functions.append(function)
            return functions
        else:
            raise TypeError("Model must be an instance of Markov or Bayesian model.")

    def get_tables(self):
        """
        Adds tables to the network.

        Example
        -------
        >>> writer = UAIWriter(model)
        >>> writer.get_tables()
        """
        if isinstance(self.model, BayesianModel):
            cpds = self.model.get_cpds()
            cpds.sort(key=lambda x: x.variable)
            tables = []
            for cpd in cpds:
                values = list(map(str, cpd.values.ravel()))
                tables.append(values)
            return tables
        elif isinstance(self.model, MarkovModel):
            factors = self.model.get_factors()
            tables = []
            for factor in factors:
                values = list(map(str, factor.values.ravel()))
                tables.append(values)
            return tables
        else:
            raise TypeError("Model must be an instance of Markov or Bayesian model.")

    def write_uai(self, filename):
        """
        Write the xml data into the file.

        Parameters
        ----------
        filename: Name of the file.

        Examples
        -------
        >>> writer = UAIWriter(model)
        >>> writer.write_xmlbif(test_file)
        """
        writer = self.__str__()
        with open(filename, 'w') as fout:
            fout.write(writer)
