# #!/usr/bin/env python3
#
# import functools
# from collections import OrderedDict
# import numpy as np
# from pgmpy import Exceptions
# # from pgmpy.Factor._factor_product import _factor_product, _factor_product_orig
# # from pgmpy.Factor._factor_product import _factor_divide
#
#
# class Factor():
#     """
#     Base class for *Factor*.
#
#     Public Methods
#     --------------
#     assignment(index)
#     get_cardinality(variable)
#     marginalize([variable_list])
#     normalise()
#     product(*factors)
#     reduce([variable_values_list])
#     """
#
#     def __init__(self, variables, cardinality, value, data=None):
#         """
#         Initialize a Factor class.
#
#         Defined above, we have the following mapping from variable
#         assignments to the index of the row vector in the value field:
#
#         +-----+-----+-----+-------------------+
#         |  x1 |  x2 |  x3 |    phi(x1, x2, x2)|
#         +-----+-----+-----+-------------------+
#         | x1_0| x2_0| x3_0|     phi.value(0)  |
#         +-----+-----+-----+-------------------+
#         | x1_0| x2_0| x3_1|     phi.value(1)  |
#         +-----+-----+-----+-------------------+
#         | x1_0| x2_1| x3_0|     phi.value(2)  |
#         +-----+-----+-----+-------------------+
#         | x1_0| x2_1| x3_1|     phi.value(3)  |
#         +-----+-----+-----+-------------------+
#         | x1_1| x2_0| x3_0|     phi.value(4)  |
#         +-----+-----+-----+-------------------+
#         | x1_1| x2_0| x3_1|     phi.value(5)  |
#         +-----+-----+-----+-------------------+
#         | x1_1| x2_1| x3_0|     phi.value(6)  |
#         +-----+-----+-----+-------------------+
#         | x1_1| x2_1| x3_1|     phi.value(7)  |
#         +-----+-----+-----+-------------------+
#
#         Parameters
#         ----------
#         variables: list
#             List of scope of factor
#         cardinality: list, array_like
#             List of cardinality of each variable
#         value: list, array_like
#             List or array of values of factor.
#             A Factor's values are stored in a row vector in the value
#             using an ordering such that the left-most variables as defined in
#             the variable field cycle through their values the fastest. More
#             concretely, for factor
#
#         Examples
#         --------
#         >>> phi = Factor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
#         """
#         self.variables = OrderedDict()
#         if len(variables) != len(cardinality):
#             raise ValueError("The size of variables and cardinality should be same")
#         for variable, card in zip(variables, cardinality):
#             self.variables[variable] = [variable + '_' + str(index)
#                                         for index in range(card)]
#         self.cardinality = np.array(cardinality)
#         self.values = np.array(value, dtype=np.double)
#         self._pos_dist = True
#         self.data = data
#         for val in self.values:
#             if val == 0:
#                 self._pos_dist = False
#                 break
#         if not self.values.shape[0] == np.product(self.cardinality):
#             raise Exceptions.SizeError("Incompetant value array")
#
#     def is_pos_dist(self):
#         """
#         Returns true if the distribution is a positive distribution
#
#         Examples
#         --------
#         >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12))
#         >>> phi.is_pos_dist()
#         True
#         """
#         return self._pos_dist
#
#     def scope(self):
#         """
#         Returns the scope of the factor.
#
#         Examples
#         --------
#         >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(8))
#         >>> phi.scope()
#         ['x1', 'x2', 'x3']
#         """
#         return list(self.variables)
#
#     def assignment(self, index):
#         """
#         Returns a list of assignments for the corresponding index.
#
#         Parameters
#         ----------
#         index: integer, list-type, ndarray
#             index or indices whose assignment is to be computed
#
#         Examples
#         --------
#         >>> from pgmpy.Factor import Factor
#         >>> import numpy as np
#         >>> phi = Factor(['diff', 'intel'], [2, 2], np.ones(4))
#         >>> phi.assignment([1, 2])
#         [['diff_0', 'intel_1'], ['diff_1', 'intel_0']]
#         """
#         if not isinstance(index, np.ndarray):
#             index = np.atleast_1d(index)
#         max_index = np.prod(self.cardinality) - 1
#         if not all(i <= max_index for i in index):
#             raise IndexError("Index greater than max possible index")
#         assignments = []
#         for ind in index:
#             assign = []
#             for card in self.cardinality[::-1]:
#                 assign.insert(0, ind % card)
#                 ind = ind/card
#             assignments.append(map(int, assign))
#         return [[self.variables[key][val] for key, val in
#                  zip(self.variables.keys(), values)] for values in assignments]
#
#     def get_variables(self):
#         """
#         Returns the variables associated with the Factor
#
#         Examples
#         --------
#         >>> from pgmpy.Factor.Factor import Factor
#         >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
#         >>> phi.get_variables()
#         ['x1', 'x2', 'x3']
#         """
#         return [var for var in self.variables.keys()]
#
#     def get_value(self, node_assignments):
#         """
#         Returns the potential value given the assignment to every node variable
#         in a list in the order of variables. The values given are in terms of
#         indexes (ith index means the i+1 th observation)
#
#         Parameters
#         ----------
#         node_assignments : list or dictionary
#             list containing the indexes of variables in order OR
#             dictionary containing the assignment of values to a number of variables
#
#         Examples
#         --------
#         >>> from pgmpy.Factor.Factor import Factor
#         >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
#         >>> phi.get_value([1,1,1])
#         9.0
#         >>> phi.get_value({'x1':1,'x2':2, 'x3':0, 'x4':2, 'x5':1})
#         5.0
#         """
#         index_for_variables = []
#         if isinstance(node_assignments, dict):
#             for var in self.variables.keys():
#                 index_for_variables.append(node_assignments[var])
#         else:
#             index_for_variables = node_assignments
#         cum_cardinality = np.cumprod(np.concatenate(([1], self.cardinality[:-1])))
#         index = np.sum(cum_cardinality * index_for_variables)
#         return self.values[index]
#
#     def get_cardinality(self, variable):
#         """
#         Returns cardinality of a given variable
#
#         Parameters
#         ----------
#         variable: string
#
#         Examples
#         --------
#         >>> from pgmpy.Factor import Factor
#         >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
#         >>> phi.get_cardinality('x1')
#         2
#         """
#         if variable not in self.variables:
#             raise Exceptions.ScopeError("%s not in scope" % variable)
#         return self.cardinality[list(self.variables.keys()).index(variable)]
#
#     def identity_factor(self):
#         """
#         Returns the identity factor.
#
#         When the identity factor of a factor is multiplied with the factor
#         it returns the factor itself.
#         """
#         return Factor(self.variables, self.cardinality, np.ones(np.product(self.cardinality)))
#
#     def reduce(self, values, inplace = True):
#         """
#         Reduces the factor to the context of given variable values.
#
#         Parameters
#         ----------
#         values: string, list-type
#             name of the variable values
#
#         Examples
#         --------
#         >>> from pgmpy.Factor import Factor
#         >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
#         >>> phi.reduce(['x1_0', 'x2_0'])
#         >>> phi.values
#         array([0., 1.])
#         """
#         if not isinstance(values, list):
#             values = [values]
#         self.data = {}
#         variables = []
#         for value in values:
#             if not '_' in value:
#                 raise TypeError("Values should be in the form of "
#                                 "variablename_index")
#             var, value_index = value.split('_')
#             variables.append(var)
#             self.data[var]=value_index
#         f = self.operations_on_variables(variables, 3, inplace)
#         self.data = None
#         return f
#
#     def marginalize(self, variables, inplace=True):
#         return self.operations_on_variables(variables, 1, inplace)
#
#     def marginalize_except(self, variables):
#         """
#         Returns marginalized factor where it marginalises everything
#         except the variables in var
#
#         Parameters
#         ----------
#         vars : string, list-type
#             Name of variables not to be marginalised
#
#         Example
#         -------
#
#         """
#         f =  self.marginalize(list(set(self.get_variables()) - set(variables)), inplace=False)
#         assert isinstance(f, Factor)
#         return f
#
#
#     def maximize(self, variables, inplace=True):
#         if self.data is None:
#             self.data = []
#             for i in range(len(self.values)):
#                 self.data.append({})
#         return self.operations_on_variables(variables, 2, inplace)
#
#     def maximize_except(self, variables):
#         f =  self.maximize(list(set(self.get_variables()) - set(variables)), inplace=False)
#         assert isinstance(f, Factor)
#         print(f.data)
#         return f
#
#     def _operations_single_variable(self, variable, op_id):
#         """
#         Returns marginalised factor for a single variable
#
#         Parameters
#         ---------
#         variable_name: string
#             name of variable to be marginalized
#
#         """
#         index = list(self.variables.keys()).index(variable)
#         cum_cardinality = (np.product(self.cardinality) / np.concatenate(([1],
#                                 np.cumprod(self.cardinality)))).astype(np.int64, copy=False)
#         #print(cum_cardinality)
#         num_elements = cum_cardinality[0]
#         sum_index = [j for i in range(0, num_elements,
#                                       cum_cardinality[index])
#                      for j in range(i, i+cum_cardinality[index+1])]
#         marg_factor = np.zeros(num_elements/self.cardinality[index])
#         new_data = None
#         if op_id == 1 :
#             for i in range(self.cardinality[index]):
#                 marg_factor += self.values[np.array(sum_index) +
#                                i*cum_cardinality[index+1]]
#         elif op_id == 2:
#             new_data = [None] * (num_elements/self.cardinality[index])
#             for i in range(int(num_elements/self.cardinality[index])):
#                 max_val = self.values[np.array(sum_index)[i] +
#                                0*cum_cardinality[index+1]]
#                 max_index = 0
#                 for j in range(self.cardinality[index]):
#                     curr_val = self.values[np.array(sum_index)[i] +
#                                j*cum_cardinality[index+1]]
#                     if(curr_val > max_val):
#                         max_val = curr_val
#                         max_index = j
#                 marg_factor[i]=max_val
#                 new_data[i] = self.data[max_index].copy()
#                 new_data[i][variable] = max_index
#         elif op_id == 3:
#             index = list(self.variables.keys()).index(variable)
#             value_index = self.data[variable]
#             marg_factor = self.values[np.array(sum_index) + int(value_index)*cum_cardinality[index+1]]
#             new_data = self.data
#
#         return  marg_factor, new_data
#
#     def operations_on_variables(self, variables,  op_id= 1, inplace = True):
#         """
#         Modifies the factor with marginalized values.
#
#         Parameters
#         ---------
#         variables: string, list-type
#             name of variable to be marginalized
#
#         Examples
#         --------
#         >>> from pgmpy.Factor.Factor import Factor
#         >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
#         >>> phi.marginalize(['x1', 'x3'])
#         >>> phi.values
#         array([ 14.,  22.,  30.])
#         >>> phi.variables
#         OrderedDict([('x2', ['x2_0', 'x2_1', 'x2_2'])])
#         """
#         if not isinstance(variables, list):
#             variables = [variables]
#         for variable in variables:
#             if variable not in self.get_variables():
#                 raise Exceptions.ScopeError("%s not in scope" % variable)
#         if not variables:
#             if inplace:
#                 return
#             else:
#                 import copy
#                 return copy.deepcopy(self)
#         ret = self
#         for variable in variables:
#             index = list(ret.variables.keys()).index(variable)
#             new_vars = ret.variables.copy()
#             del(new_vars[variable])
#             values_data_tuple = ret._operations_single_variable(variable, op_id)
#             ret = Factor(new_vars, np.delete(ret.cardinality, index), values_data_tuple[0], values_data_tuple[1] )
#         if inplace:
#             self.variables = ret.variables
#             self.values = ret.values
#             self.cardinality = ret.cardinality
#             self.data = ret.data
#         else:
#             return ret
#
#     def normalize(self):
#         """
#         Normalizes the values of factor so that they sum to 1.
#         """
#         self.values = self.values / np.sum(self.values)
#
#
#
#     def product(self, *factors):
#         """
#         Returns the factor product with factors.
#
#         Parameters
#         ----------
#         *factors: Factor1, Factor2, ...
#             Factors to be multiplied
#
#         Example
#         -------
#         >>> from pgmpy.Factor import Factor
#         >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
#         >>> phi2 = Factor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
#         >>> phi = phi1.product(phi2)
#         >>> phi.variables
#         OrderedDict([('x1', ['x1_0', 'x1_1']), ('x2', ['x2_0', 'x2_1', 'x2_2']),
#                 ('x3', ['x3_0', 'x3_1']), ('x4', ['x4_0', 'x4_1'])])
#         """
#         return factor_product(self, *factors)
#
#     def divide(self, factor):
#         """
#         Returns factor division of two factors
#
#         Parameters
#         ----------
#         factor : Factor
#             The denominator
#
#         Examples
#         --------
#         >>> from pgmpy.Factor.Factor import Factor
#         >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
#         >>> phi2 = Factor(['x3', 'x1'], [2, 2], [x+1 for x in range(4)])
#         >>> phi = phi1.divide(phi2)
#         >>> phi
#         x1	x2	x3	phi(x1, x2, x3)
#         x1_0	x2_0	x3_0	0.0
#         x1_1	x2_0	x3_0	0.5
#         x1_0	x2_1	x3_0	0.666666666667
#         x1_1	x2_1	x3_0	0.75
#         x1_0	x2_2	x3_0	4.0
#         x1_1	x2_2	x3_0	2.5
#         x1_0	x2_0	x3_1	2.0
#         x1_1	x2_0	x3_1	1.75
#         x1_0	x2_1	x3_1	8.0
#         x1_1	x2_1	x3_1	4.5
#         x1_0	x2_2	x3_1	3.33333333333
#         x1_1	x2_2	x3_1	2.75
#         """
#         if factor.is_pos_dist():
#             return factor_divide(self, factor)
#         else:
#             raise ValueError("Division not possible : The second distribution" +
#                              " is not a positive distribution")
#
#     def sum_values(self):
#         """
#         Returns the sum of potentials
#
#         Examples
#         --------
#         >>> from pgmpy.Factor.Factor import Factor
#         >>> phi1 = Factor(['x1','x2',], [2,2],[1,2,2,4] )
#         >>> phi1.sum_values()
#         9.0
#         """
#         return np.sum(self.values)
#
#     def __str__(self):
#         return self._str('phi')
#
#     __repr__ = __str__
#
#     def _str(self, phi_or_p):
#         string = ""
#         for var in self.variables:
#             string += str(var) + "\t"
#         string += phi_or_p + '(' + ', '.join(self.variables) + ')'
#         string += "\n"
#
#         #fun and gen are functions to generate the different values of variables in the table.
#         #gen starts with giving fun initial value of b=[0, 0, 0] then fun tries to increment it
#         #by 1.
#         def fun(b, index=len(self.cardinality)-1):
#             b[index] += 1
#             if b[index] == self.cardinality[index]:
#                 b[index] = 0
#                 fun(b, index -1)
#             return b
#
#         def gen():
#             b = [0] * len(self.variables)
#             yield b
#             for i in range(np.prod(self.cardinality) - 1):
#                 yield fun(b)
#
#         value_index = 0
#         for prob in gen():
#             prob_list = [list(self.variables)[i] + '_' + str(prob[i]) for i in range(len(self.variables))]
#             string += '\t'.join(prob_list) + '\t' + str(self.values[value_index]) + '\n'
#             value_index += 1
#         return string[:-1]
#
#     def __mul__(self, other):
#         return self.product(other)
#
#     def __eq__(self, other):
#         if type(self) != type(other):
#             return False
#
#         if self.variables == other.variables and all(self.cardinality == other.cardinality) \
#                 and all(self.values == other.values):
#             return True
#         else:
#             return False
#
#     def __hash__(self):
#         """
#         Returns the hash of the factor object based on the scope of the factor.
#         """
#         return hash(' '.join(self.variables) + ' '.join(map(str, self.cardinality)) +
#                     ' '.join(list(map(str, self.values))))
#
#
# def _bivar_factor_divide(phi1, phi2):
#     """
#     Returns phi1 divided by phi2
#
#     Parameters
#     ----------
#     phi1: Factor
#         Numerator
#     phi2: Factor
#         Denominator
#
#     See Also
#     --------
#     factor_divide
#     """
#     phi1_vars = list(phi1.variables.keys())
#     phi2_vars = list(phi2.variables.keys())
#     if set(phi2_vars) - set(phi1_vars):
#         raise ValueError("The vars in phi2 are not a subset of vars in phi1")
#     common_var_list = phi2_vars
#     common_var_index_list = np.array([[phi1_vars.index(var), phi2_vars.index(var)]
#                                       for var in common_var_list])
#     common_card_product = np.prod([phi1.cardinality[index[0]] for index
#                                    in common_var_index_list])
#     size = np.prod(phi1.cardinality)
#     product = _factor_divide(phi1.values,
#                              phi2.values,
#                              size,
#                              common_var_index_list,
#                              phi1.cardinality,
#                              phi2.cardinality)
#     variables = phi1_vars
#     cardinality = list(phi1.cardinality)
#     phi = Factor(variables, cardinality, product)
#     return phi
#
# def cum_card(phi1):
#     return (np.product(phi1.cardinality) / np.cumprod(phi1.cardinality)).astype(np.int64, copy=False)
#
# def ref(variables, vars1):
#     l=[]
#     for var in variables:
#         if var in vars1:
#             l.append(vars1.index(var))
#         else:
#             l.append(-1)
#     return np.array(l)
#
#
# def _bivar_factor_product(phi1, phi2):
#     """
#     Returns product of two factors.
#
#     Parameters
#     ----------
#     phi1: Factor
#
#     phi2: Factor
#
#     See Also
#     --------
#     factor_product
#     """
#     vars1 = list(phi1.variables.keys())
#     vars2 = list(phi2.variables.keys())
#     common_var_list = [var1 for var1 in vars1 for var2 in vars2
#                        if var1 == var2]
#     if common_var_list:
#         variables = [var for var in vars1]
#         variables.extend(var for var in phi2.variables
#                          if var not in common_var_list)
#         cardinality = list(phi1.cardinality)
#         cardinality.extend(phi2.get_cardinality(var) for var in phi2.variables
#                            if var not in common_var_list)
#         size = np.prod(cardinality)
#         cum_card_1 = cum_card(phi1)
#         cum_card_2 = cum_card(phi2)
#         ref_1 = ref(variables, vars1)
#         ref_2 = ref(variables, vars2)
#         product = _factor_product(np.array(cardinality), size,
#                                   phi1.values, cum_card_1, ref_1,
#                                   phi2.values, cum_card_2, ref_2)
#         phi = Factor(variables, cardinality, product)
#         return phi
#     else:
#         size = np.prod(phi1.cardinality) * np.prod(phi2.cardinality)
#         product = _factor_product(phi1.values,
#                                   phi2.values,
#                                   size)
#         variables = vars1 + vars2
#         cardinality = list(phi1.cardinality) + list(phi2.cardinality)
#         phi = Factor(variables, cardinality, product)
#         return phi
#
#
#
# def _bivar_factor_product_orig(phi1, phi2):
#     """
#     Returns product of two factors.
#
#     Parameters
#     ----------
#     phi1: Factor
#
#     phi2: Factor
#
#     See Also
#     --------
#     factor_product
#     """
#     vars1 = list(phi1.variables.keys())
#     vars2 = list(phi2.variables.keys())
#     common_var_list = [var1 for var1 in vars1 for var2 in vars2
#                        if var1 == var2]
#     if common_var_list:
#         common_var_index_list = np.array([[vars1.index(var), vars2.index(var)]
#                                           for var in common_var_list])
#         common_card_product = np.prod([phi1.cardinality[index[0]] for index
#                                        in common_var_index_list])
#         size = np.prod(phi1.cardinality) * np.prod(
#             phi2.cardinality) / common_card_product
#         product = _factor_product_orig(phi1.values,
#                                   phi2.values,
#                                   size,
#                                   common_var_index_list,
#                                   phi1.cardinality,
#                                   phi2.cardinality)
#         variables = vars1
#         variables.extend(var for var in phi2.variables
#                          if var not in common_var_list)
#         cardinality = list(phi1.cardinality)
#         cardinality.extend(phi2.get_cardinality(var) for var in phi2.variables
#                            if var not in common_var_list)
#         phi = Factor(variables, cardinality, product)
#         return phi
#     else:
#         size = np.prod(phi1.cardinality) * np.prod(phi2.cardinality)
#         product = _factor_product_orig(phi1.values,
#                                   phi2.values,
#                                   size)
#         variables = vars1 + vars2
#         cardinality = list(phi1.cardinality) + list(phi2.cardinality)
#         phi = Factor(variables, cardinality, product)
#         return phi
#
#
# # def factor_product(*args):
# #     """
# #     Returns factor product of multiple factors.
# #
# #     Parameters
# #     ----------
# #     factor1, factor2, .....: Factor
# #         factors to be multiplied
# #
# #     Examples
# #     --------
# #     >>> from pgmpy.Factor.Factor import Factor, factor_divide
# #     >>> phi1 = Factor(['x1','x2',], [2,2],[1,2,2,4] )
# #     >>> phi2 = Factor(['x1'], [2], [1,2])
# #     >>> phi = factor_divide(phi1, phi2)
# #     >>> phi
# #     x1	x2	phi(x1, x2)
# #     x1_0	x2_0	1.0
# #     x1_1	x2_0	1.0
# #     x1_0	x2_1	2.0
# #     x1_1	x2_1	2.0
# #     >>>
# #     """
# #     if not all(isinstance(phi, Factor) for phi in args):
# #         raise TypeError("Input parameters must be factors")
# #     return functools.reduce(_bivar_factor_product, args)
# # <<<<<<< HEAD
# # =======
#
#
# def factor_divide(factor1, factor2):
#     """
#     Returns factor division of two factors
#
#     Parameters
#     ----------
#     factor1: Factor
#         The numerator
#     factor2 : Factor
#         The denominator
#
#     Examples
#     --------
#     >>> from pgmpy.Factor.Factor import Factor, factor_divide
#     >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
#     >>> phi2 = Factor(['x3', 'x1'], [2, 2], [x+1 for x in range(4)])
#     >>> assert isinstance(phi1, Factor)
#     >>> assert isinstance(phi2, Factor)
#     >>> phi = factor_divide(phi1, phi2)
#     >>> phi
#     x1	x2	x3	phi(x1, x2, x3)
#     x1_0	x2_0	x3_0	0.0
#     x1_1	x2_0	x3_0	0.5
#     x1_0	x2_1	x3_0	0.666666666667
#     x1_1	x2_1	x3_0	0.75
#     x1_0	x2_2	x3_0	4.0
#     x1_1	x2_2	x3_0	2.5
#     x1_0	x2_0	x3_1	2.0
#     x1_1	x2_0	x3_1	1.75
#     x1_0	x2_1	x3_1	8.0
#     x1_1	x2_1	x3_1	4.5
#     x1_0	x2_2	x3_1	3.33333333333
#     x1_1	x2_2	x3_1	2.75
#     """
#     if not (isinstance(factor1, Factor) and isinstance(factor2, Factor)):
#         raise TypeError("Input parameters must be factors")
#     return _bivar_factor_divide(factor1, factor2)



import functools
from collections import OrderedDict
import numpy as np
from pgmpy.Exceptions import Exceptions
from pgmpy.Factor._factor_product import _factor_product, _factor_divide


class Factor():
    """
    Base class for *Factor*.

    Public Methods
    --------------
    assignment(index)
    get_cardinality(variable)
    marginalize([variable_list])
    normalise()
    product(*factors)
    reduce([variable_values_list])
    """

    def __init__(self, variables, cardinality, value, data=None):
        """
        Initialize a Factor class.

        Defined above, we have the following mapping from variable
        assignments to the index of the row vector in the value field:

        +-----+-----+-----+-------------------+
        |  x1 |  x2 |  x3 |    phi(x1, x2, x2)|
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_0|     phi.value(0)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_1|     phi.value(1)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_0|     phi.value(2)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_1|     phi.value(3)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_0|     phi.value(4)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_1|     phi.value(5)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_0|     phi.value(6)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_1|     phi.value(7)  |
        +-----+-----+-----+-------------------+

        Parameters
        ----------
        variables: list
            List of scope of factor
        cardinality: list, array_like
            List of cardinality of each variable
        value: list, array_like
            List or array of values of factor.
            A Factor's values are stored in a row vector in the value
            using an ordering such that the left-most variables as defined in
            the variable field cycle through their values the fastest. More
            concretely, for factor

        Examples
        --------
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
        """
        self.variables = OrderedDict()
        if len(variables) != len(cardinality):
            raise ValueError("The size of variables and cardinality should be same")
        for variable, card in zip(variables, cardinality):
            self.variables[variable] = [variable + '_' + str(index)
                                        for index in range(card)]
        self.cardinality = np.array(cardinality)
        self.values = np.array(value, dtype=np.double)
        self._pos_dist = True
        self.data = data
        for val in self.values:
            if val == 0:
                self._pos_dist = False
                break
        if not self.values.shape[0] == np.product(self.cardinality):
            raise Exceptions.SizeError("Incompetant value array")

    def is_pos_dist(self):
        """
        Returns true if the distribution is a positive distribution

        Examples
        --------
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12))
        >>> phi.is_pos_dist()
        True
        """
        return self._pos_dist

    def singleton_factor(self):
        return len(self.get_variables()) == 1

    def pairwise_submodular_factor(self):
        return len(self.get_variables()) == 2 and \
            self.get_value([1, 1]) * self.get_value([0, 0]) >= \
            self.get_value([1, 0]) * self.get_value([0, 1])

    def scope(self):
        """
        Returns the scope of the factor.

        Examples
        --------
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(8))
        >>> phi.scope()
        ['x1', 'x2', 'x3']
        """
        return list(self.variables)

    def assignment(self, index):
        """
        Returns a list of assignments for the corresponding index.

        Parameters
        ----------
        index: integer, list-type, ndarray
            index or indices whose assignment is to be computed

        Examples
        --------
        >>> from pgmpy.Factor import Factor
        >>> import numpy as np
        >>> phi = Factor(['diff', 'intel'], [2, 2], np.ones(4))
        >>> phi.assignment([1, 2])
        [['diff_0', 'intel_1'], ['diff_1', 'intel_0']]
        """
        if not isinstance(index, np.ndarray):
            index = np.atleast_1d(index)
        max_index = np.prod(self.cardinality) - 1
        if not all(i <= max_index for i in index):
            raise IndexError("Index greater than max possible index")
        assignments = []
        for ind in index:
            assign = []
            for card in self.cardinality[::-1]:
                assign.insert(0, ind % card)
                ind = ind / card
            assignments.append(map(int, assign))
        return [[self.variables[key][val] for key, val in
                 zip(self.variables.keys(), values)] for values in assignments]

    def get_variables(self):
        """
        Returns the variables associated with the Factor

        Examples
        --------
        >>> from pgmpy.Factor.Factor import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.get_variables()
        ['x1', 'x2', 'x3']
        """
        return [var for var in self.variables.keys()]

    def get_log_value(self, node_assignments):
        import math

        return math.log(self.get_value(node_assignments))

    def get_value(self, node_assignments):
        """
        Returns the potential value given the assignment to every node variable
        in a list in the order of variables. The values given are in terms of
        indexes (ith index means the i+1 th observation)

        Parameters
        ----------
        node_assignments : list or dictionary
            list containing the indexes of variables in order OR
            dictionary containing the assignment of values to a number of variables

        Examples
        --------
        >>> from pgmpy.Factor.Factor import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.get_value([1,1,1])
        9.0
        >>> phi.get_value({'x1':1,'x2':2, 'x3':0, 'x4':2, 'x5':1})
        5.0
        """
        index_for_variables = []
        if isinstance(node_assignments, dict):
            for var in self.variables.keys():
                index_for_variables.append(node_assignments[var])
        else:
            index_for_variables = node_assignments
        cum_cardinality = np.cumprod(np.concatenate(([1], self.cardinality[:-1])))
        index = np.sum(cum_cardinality * index_for_variables)
        return self.values[index]

    def get_cardinality(self, variable):
        """
        Returns cardinality of a given variable

        Parameters
        ----------
        variable: string

        Examples
        --------
        >>> from pgmpy.Factor import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.get_cardinality('x1')
        2
        """
        if variable not in self.variables:
            raise Exceptions.ScopeError("%s not in scope" % variable)
        return self.cardinality[list(self.variables.keys()).index(variable)]

    def identity_factor(self):
        """
        Returns the identity factor.

        When the identity factor of a factor is multiplied with the factor
        it returns the factor itself.
        """
        return Factor(self.variables, self.cardinality, np.ones(np.product(self.cardinality)))

    def reduce(self, values, inplace=True):
        """
        Reduces the factor to the context of given variable values.

        Parameters
        ----------
        values: string, list-type
            name of the variable values

        Examples
        --------
        >>> from pgmpy.Factor import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.reduce(['x1_0', 'x2_0'])
        >>> phi.values
        array([0., 1.])
        """
        if not isinstance(values, list):
            values = [values]
        self.data = {}
        variables = []
        for value in values:
            if not '_' in value:
                raise TypeError("Values should be in the form of "
                                "variablename_index")
            var, value_index = value.split('_')
            variables.append(var)
            self.data[var] = int(value_index)
        f = self.operations_on_variables(variables, 3, inplace)
        self.data = None
        return f

    def marginalize(self, variables, inplace=True):
        return self.operations_on_variables(variables, 1, inplace)

    def marginalize_except(self, variables):
        """
        Returns marginalized factor where it marginalises everything
        except the variables in var

        Parameters
        ----------
        vars : string, list-type
            Name of variables not to be marginalised

        Example
        -------

        """
        f = self.marginalize(list(set(self.get_variables()) - set(variables)), inplace=False)
        assert isinstance(f, Factor)
        return f

    def maximize(self, variables, inplace=True):
        if self.data is None:
            self.data = []
            for i in range(len(self.values)):
                self.data.append([])
        return self.operations_on_variables(variables, 2, inplace)

    def maximize_except(self, variables):
        f = self.maximize(list(set(self.get_variables()) - set(variables)), inplace=False)
        assert isinstance(f, Factor)
        return f

    def _operations_single_variable(self, variable, op_id):
        """
        Returns marginalised factor for a single variable

        Parameters
        ---------
        variable_name: string
            name of variable to be marginalized

        """
        index = list(self.variables.keys()).index(variable)
        cum_cardinality = (np.product(self.cardinality) /
                           np.concatenate(([1], np.cumprod(self.cardinality)))).astype(np.int64, copy=False)
        #print(cum_cardinality)
        num_elements = cum_cardinality[0]
        sum_index = [j for i in range(0, num_elements,
                                      cum_cardinality[index])
                     for j in range(i, i + cum_cardinality[index + 1])]
        marg_factor = np.zeros(num_elements / self.cardinality[index])
        new_data = None
        if op_id == 1:
            for i in range(self.cardinality[index]):
                marg_factor += self.values[np.array(sum_index) +
                                           i * cum_cardinality[index + 1]]
        elif op_id == 2:
            new_data = [None] * (num_elements / self.cardinality[index])
            for i in range(int(num_elements / self.cardinality[index])):
                max_val = self.values[np.array(sum_index)[i] +
                                      0 * cum_cardinality[index + 1]]
                max_index = 0
                for j in range(self.cardinality[index]):
                    curr_val = self.values[np.array(sum_index)[i] +
                                           j * cum_cardinality[index + 1]]
                    if curr_val > max_val:
                        max_val = curr_val
                        max_index = j
                marg_factor[i] = max_val
                new_data[i] = [tup for tup in self.data[np.array(sum_index)[i] +
                                                        max_index * cum_cardinality[index + 1]]]
                new_data[i].append((variable, max_index))
        elif op_id == 3:
            index = list(self.variables.keys()).index(variable)
            value_index = self.data[variable]
            if value_index < 0 or value_index >= self.get_cardinality(variable):
                raise ValueError("The value for the variable " + variable + "is out of bound")
            marg_factor = self.values[np.array(sum_index) + int(value_index) * cum_cardinality[index + 1]]
            new_data = self.data

        return marg_factor, new_data

    def operations_on_variables(self, variables, op_id=1, inplace=True):
        """
        Modifies the factor with marginalized values.

        Parameters
        ---------
        variables: string, list-type
            name of variable to be marginalized

        Examples
        --------
        >>> from pgmpy.Factor.Factor import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.marginalize(['x1', 'x3'])
        >>> phi.values
        array([ 14.,  22.,  30.])
        >>> phi.variables
        OrderedDict([('x2', ['x2_0', 'x2_1', 'x2_2'])])
        """
        if not isinstance(variables, list):
            variables = [variables]
        for variable in variables:
            if variable not in self.get_variables():
                raise Exceptions.ScopeError("%s not in scope" % variable)
        if not variables:
            if inplace:
                return
            else:
                import copy

                return copy.deepcopy(self)
        ret = self
        for variable in variables:
            index = list(ret.variables.keys()).index(variable)
            new_vars = ret.variables.copy()
            del (new_vars[variable])
            values_data_tuple = ret._operations_single_variable(variable, op_id)
            ret = Factor(new_vars, np.delete(ret.cardinality, index), values_data_tuple[0], values_data_tuple[1])
        if inplace:
            self.variables = ret.variables
            self.values = ret.values
            self.cardinality = ret.cardinality
            self.data = ret.data
        else:
            return ret

    def normalize(self):
        """
        Normalizes the values of factor so that they sum to 1.
        """
        self.values = self.values / np.sum(self.values)

    def product(self, *factors):
        """
        Returns the factor product with factors.

        Parameters
        ----------
        *factors: Factor1, Factor2, ...
            Factors to be multiplied

        Example
        -------
        >>> from pgmpy.Factor import Factor
        >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi2 = Factor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
        >>> phi = phi1.product(phi2)
        >>> phi.variables
        OrderedDict([('x1', ['x1_0', 'x1_1']), ('x2', ['x2_0', 'x2_1', 'x2_2']),
                ('x3', ['x3_0', 'x3_1']), ('x4', ['x4_0', 'x4_1'])])
        """
        return factor_product(self, *factors)

    def divide(self, factor):
        """
        Returns factor division of two factors

        Parameters
        ----------
        factor : Factor
            The denominator

        Examples
        --------
        >>> from pgmpy.Factor.Factor import Factor
        >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi2 = Factor(['x3', 'x1'], [2, 2], [x+1 for x in range(4)])
        >>> phi = phi1.divide(phi2)
        >>> phi
        x1	x2	x3	phi(x1, x2, x3)
        x1_0	x2_0	x3_0	0.0
        x1_1	x2_0	x3_0	0.5
        x1_0	x2_1	x3_0	0.666666666667
        x1_1	x2_1	x3_0	0.75
        x1_0	x2_2	x3_0	4.0
        x1_1	x2_2	x3_0	2.5
        x1_0	x2_0	x3_1	2.0
        x1_1	x2_0	x3_1	1.75
        x1_0	x2_1	x3_1	8.0
        x1_1	x2_1	x3_1	4.5
        x1_0	x2_2	x3_1	3.33333333333
        x1_1	x2_2	x3_1	2.75
        """
        if factor.is_pos_dist():
            return factor_divide(self, factor)
        else:
            raise ValueError("Division not possible : The second distribution" +
                             " is not a positive distribution")

    def sum_values(self):
        """
        Returns the sum of potentials

        Examples
        --------
        >>> from pgmpy.Factor.Factor import Factor
        >>> phi1 = Factor(['x1','x2',], [2,2],[1,2,2,4] )
        >>> phi1.sum_values()
        9.0
        """
        return np.sum(self.values)

    def __str__(self):
        return self._str('phi')

    def _str(self, phi_or_p):
        string = ""
        for var in self.variables:
            string += str(var) + "\t\t"
        string += phi_or_p + '(' + ', '.join(self.variables) + ')'
        string += "\n"
        string += '-' * 2 * len(string) + '\n'

        #fun and gen are functions to generate the different values of variables in the table.
        #gen starts with giving fun initial value of b=[0, 0, 0] then fun tries to increment it
        #by 1.
        def fun(b, index=len(self.cardinality) - 1):
            b[index] += 1
            if b[index] == self.cardinality[index]:
                b[index] = 0
                fun(b, index - 1)
            return b

        def gen():
            b = [0] * len(self.variables)
            yield b
            for i in range(np.prod(self.cardinality) - 1):
                yield fun(b)

        value_index = 0
        for prob in gen():
            prob_list = [list(self.variables)[i] + '_' + str(prob[i]) for i in range(len(self.variables))]
            string += '\t\t'.join(prob_list) + '\t\t' + str(self.values[value_index])
            if self.data is not None:
                string += '\t' + str(self.data[value_index])
            string += '\n'
            value_index += 1
        return string[:-1]

    def __mul__(self, other):
        return self.product(other)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        if self.variables == other.variables and all(self.cardinality == other.cardinality) \
                and all(self.values == other.values):
            return True
        else:
            return False

    def __hash__(self):
        """
        Returns the hash of the factor object based on the scope of the factor.
        """
        return hash(' '.join(self.variables) + ' '.join(map(str, self.cardinality)) +
                    ' '.join(list(map(str, self.values))))


def _bivar_factor_divide(phi1, phi2):
    """
    Returns phi1 divided by phi2

    Parameters
    ----------
    phi1: Factor
        Numerator
    phi2: Factor
        Denominator

    See Also
    --------
    factor_divide
    """
    phi1_vars = list(phi1.variables.keys())
    phi2_vars = list(phi2.variables.keys())
    if set(phi2_vars) - set(phi1_vars):
        raise ValueError("The vars in phi2 are not a subset of vars in phi1")
    common_var_list = phi2_vars
    common_var_index_list = np.array([[phi1_vars.index(var), phi2_vars.index(var)]
                                      for var in common_var_list])
    #common_card_product = np.prod([phi1.cardinality[index[0]] for index
    #                               in common_var_index_list])
    size = np.prod(phi1.cardinality)
    product = _factor_divide(phi1.values,
                             phi2.values,
                             size,
                             common_var_index_list,
                             phi1.cardinality,
                             phi2.cardinality)
    variables = phi1_vars
    cardinality = list(phi1.cardinality)
    phi = Factor(variables, cardinality, product)
    return phi


def cum_card(phi1):
    return (np.product(phi1.cardinality) / np.cumprod(phi1.cardinality)).astype(np.int64, copy=False)


def ref(variables, vars1):
    l = []
    for var in variables:
        if var in vars1:
            l.append(vars1.index(var))
        else:
            l.append(-1)
    return np.array(l)


def _bivar_factor_product(phi1, phi2):
    """
    Returns product of two factors.

    Parameters
    ----------
    phi1: Factor

    phi2: Factor

    See Also
    --------
    factor_product
    """
    vars1 = list(phi1.variables.keys())
    vars2 = list(phi2.variables.keys())
    common_var_list = [var1 for var1 in vars1 for var2 in vars2
                       if var1 == var2]

    variables = [var for var in vars1]
    variables.extend(var for var in phi2.variables
                     if var not in common_var_list)
    cardinality = list(phi1.cardinality)
    cardinality.extend(phi2.get_cardinality(var) for var in phi2.variables
                       if var not in common_var_list)
    size = np.prod(cardinality)
    cum_card_1 = cum_card(phi1)
    cum_card_2 = cum_card(phi2)
    ref_1 = ref(variables, vars1)
    ref_2 = ref(variables, vars2)
    product, new_data = _factor_product(np.array(cardinality), size,
                                        phi1.values, cum_card_1, ref_1,
                                        phi2.values, cum_card_2, ref_2,
                                        phi1.data, phi2.data)
    phi = Factor(variables, cardinality, product, new_data)
    return phi


def _factor_product_python_version(card_prod, size,
                                   x, card_x, ref_x,
                                   y, card_y, ref_y):
    product_arr = np.zeros(size)
    prod_indices = np.array([0] * card_prod.shape[0], dtype=np.int)
    x_index = 0
    y_index = 0
    for i in range(size):
        product_arr[i] = x[x_index] * y[y_index]
        j = card_prod.shape[0] - 1
        flag = 0
        while True:
            old_value = prod_indices[j]
            prod_indices[j] += 1
            if prod_indices[j] == card_prod[j]:
                prod_indices[j] = 0
            else:
                flag = 1
            if ref_x[j] != -1:
                x_index += (prod_indices[j] - old_value) * card_x[ref_x[j]]
            if ref_y[j] != -1:
                y_index += (prod_indices[j] - old_value) * card_y[ref_y[j]]
            j -= 1
            if flag == 1:
                break
    return product_arr


def _bivar_factor_product_orig(phi1, phi2):
    """
    Returns product of two factors.

    Parameters
    ----------
    phi1: Factor

    phi2: Factor

    See Also
    --------
    factor_product
    """
    phi1_vars = list(phi1.variables)
    phi2_vars = list(phi2.variables)
    common_var_list = [var for var in phi1_vars if var in phi2_vars]
    if common_var_list:
        common_var_index_list = np.array([[phi1_vars.index(var), phi2_vars.index(var)]
                                          for var in common_var_list])
        common_card_product = np.prod([phi1.cardinality[index[0]] for index
                                       in common_var_index_list])

        variables = phi1_vars
        variables.extend([var for var in phi2.variables
                         if var not in common_var_list])
        cardinality = list(phi1.cardinality)
        cardinality.extend(phi2.get_cardinality(var) for var in phi2.variables
                           if var not in common_var_list)

        phi1_indexes = [i for i in range(len(phi1.variables))]
        phi2_indexes = [variables.index(var) for var in phi2.variables]
        values = []
        phi1_cumprod = np.delete(np.concatenate((np.array([1]), np.cumprod(phi1.cardinality[::-1])), axis=1)[::-1], 0)
        phi2_cumprod = np.delete(np.concatenate((np.array([1]), np.cumprod(phi2.cardinality[::-1])), axis=1)[::-1], 0)
        from itertools import product
        for index in product(*[range(card) for card in cardinality]):
            index = np.array(index)
            values.append(phi1.values[np.sum(index[phi1_indexes] * phi1_cumprod)] * phi2.values[np.sum(index[phi2_indexes] * phi2_cumprod)])

        phi = Factor(variables, cardinality, values)
        return phi
    else:
        values = np.array([])
        for value in phi1.values:
            values = np.concatenate((values, value*phi2.values), axis=1)
        variables = phi1_vars + phi2_vars
        cardinality = list(phi1.cardinality) + list(phi2.cardinality)
        phi = Factor(variables, cardinality, values)
        return phi

def factor_product(*args):
    """
    Returns factor product of multiple factors.

    Parameters
    ----------
    factor1, factor2, .....: Factor
        factors to be multiplied

    Examples
    --------
    >>> from pgmpy.Factor.Factor import Factor, factor_divide
    >>> phi1 = Factor(['x1','x2',], [2,2],[1,2,2,4] )
    >>> phi2 = Factor(['x1'], [2], [1,2])
    >>> phi = factor_divide(phi1, phi2)
    >>> phi
    x1	x2	phi(x1, x2)
    x1_0	x2_0	1.0
    x1_1	x2_0	1.0
    x1_0	x2_1	2.0
    x1_1	x2_1	2.0
    >>>
    """
    if not all(isinstance(phi, Factor) for phi in args):
        raise TypeError("Input parameters must be factors")
    return functools.reduce(_bivar_factor_product, args)

def factor_divide(factor1, factor2):
    """
    Returns factor division of two factors

    Parameters
    ----------
    factor1: Factor
        The numerator
    factor2 : Factor
        The denominator

    Examples
    --------
    >>> from pgmpy.Factor.Factor import Factor, factor_divide
    >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
    >>> phi2 = Factor(['x3', 'x1'], [2, 2], [x+1 for x in range(4)])
    >>> assert isinstance(phi1, Factor)
    >>> assert isinstance(phi2, Factor)
    >>> phi = factor_divide(phi1, phi2)
    >>> phi
    x1	x2	x3	phi(x1, x2, x3)
    x1_0	x2_0	x3_0	0.0
    x1_1	x2_0	x3_0	0.5
    x1_0	x2_1	x3_0	0.666666666667
    x1_1	x2_1	x3_0	0.75
    x1_0	x2_2	x3_0	4.0
    x1_1	x2_2	x3_0	2.5
    x1_0	x2_0	x3_1	2.0
    x1_1	x2_0	x3_1	1.75
    x1_0	x2_1	x3_1	8.0
    x1_1	x2_1	x3_1	4.5
    x1_0	x2_2	x3_1	3.33333333333
    x1_1	x2_2	x3_1	2.75
    """
    if not (isinstance(factor1, Factor) and isinstance(factor2, Factor)):
        raise TypeError("Input parameters must be factors")
    return _bivar_factor_divide(factor1, factor2)
