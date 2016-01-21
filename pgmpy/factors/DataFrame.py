import numpy as np

from pgmpy.extern import six

class DataFrame(object):

	#TODO: add docstring with examples
    def __init__(self, variables=[], values=[]):
        """
        Initializes a DataFrame Class.

        Parameters
        ----------
        variables: list, array-like
        List of variables whose data is given in the sample.

        values: list, array-like
        List of values in the sample.
        """
        values = np.asarray(values)

        if isinstance(variables, six.string_types):
            raise TypeError("Variables: Expected type list or array like, got string")
        
        self.values = np.asarray(values)

        if values.dtype != int:
            raise TypeError("Values: Expected type int, got ", values.dtype)

        if values.size%len(variables):
            raise ValueError("Values: Number of samples for each variable should be same.")

        self.variables = list(variables)
        self.values = values.reshape(-1, len(variables))
