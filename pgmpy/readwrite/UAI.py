from itertools import combinations


from pgmpy.models import BayesianModel, MarkovModel
from pgmpy.factors import TabularCPD, Factor

import numpy as np


class UAIReader:
    def __init__(self, path, var_names=None):
        self.file = open(path)

        model = self.file.readline().strip()
        if model == 'MARKOV':
            self.model = MarkovModel()
        elif model == 'BAYES':
            self.model = BayesianModel()

        self.no_of_nodes = int(self.file.readline().strip())
        if var_names and len(var_names) == self.no_of_nodes:
            self.variables = np.array(var_names)
            self.model.add_nodes_from(var_names)
        else:
            self.variables = np.array(list(map(str, range(self.no_of_nodes))))
            self.model.add_nodes_from(list(map(str, range(self.no_of_nodes))))

        self.cardinality = np.array(list(map(int, self.file.readline().strip().split(' '))))

        self.no_of_functions = int(self.file.readline().strip())

        self.functions_data = []
        self.functions = []

        for i in range(self.no_of_functions):
            self.functions_data.append(self.file.readline().strip())

        self.file.readline()

        for i in range(self.no_of_functions):
            function_data = self.functions_data[i]
            vars_index = list(map(int, function_data.strip().split()))[1:]
            function_vars = self.variables[vars_index]

            self.model.add_edges_from(combinations(function_vars, 2))

            if model == 'MARKOV':
                no_of_params = self.file.readline().strip()
                params = []
                while True:
                    line = self.file.readline().strip()
                    if line == '':
                        break
                    else:
                        params.extend(map(float, line.split(' ')))

                factor = Factor(variables=function_vars, cardinality=self.cardinality[vars_index],
                                value=params)
                self.functions.append(factor)

        self.model.add_factors(*self.functions)

    def get_model(self):
        return self.model
