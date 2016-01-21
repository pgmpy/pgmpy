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
        
        self.values = values

        if values.dtype != int:
            raise TypeError("Values: Expected type int, got ", values.dtype)

        if values.size%len(variables):
            raise ValueError("Values: Number of samples for each variable should be same.")

        self.variables = list(variables)
        self.values = values.reshape(-1, len(variables))

    def get_variables(self):
        """
        Returns the list of variables whose data is given in the samples.

        Returns
        -------
        list: List of variable names.

        Examples
        --------
        >>> from pgmpy.factors import DataFrame
        >>> df = DataFrame(['A', 'B', 'C'], [1,1,1,0,0,1,1,1,0,1,0,1])
        >>> df.get_variables()
        ['A', 'B', 'C']
        """
        return self.variables

    def get_values(self):
        """
        Returns the list of values in the given samples.

        Returns
        -------
        ndarray: 2-D Array of values with number of columns equal to the
        number of variables and number of rows equal to the number of 
        samples.

        Examples
        --------
        >>> from pgmpy.factors import DataFrame
        >>> df = DataFrame(['A', 'B', 'C'], [1,1,1,0,0,1,1,1,0,1,0,1])
        >>> df.get_values()
        array([[1, 1, 1],
               [0, 0, 1],
               [1, 1, 0],
               [1, 0, 1]])
        """
        return self.values
