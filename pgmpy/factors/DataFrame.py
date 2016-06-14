import numpy as np

from pgmpy.extern import six
from pgmpy.extern import tabulate


class DataFrame(object):
    """
    Base class for DataFrame objects.
    A DataFrame object comprises a list of variables and a
    2D numpy array whose columns represent values of the corresponding
    variables and rows represent samples.

    Public Methods
    --------------
    get_variables()
    get_values()
    get_num_of_samples()

    Examples
    --------
    >>> from pgmpy.factors import DataFrame
    >>> df1 = DataFrame(['A', 'B', 'C'], [0, 1, 2, 1, 1, 1, 2, 3, 0])
    >>> df2 = DataFrame(['A', 'B', 'C'], np.array([[0, 0, 0],
                                                   [0, 1, 1],
                                                   [0, 1, 1],
                                                   [1, 2, 3],
                                                   [0, 2, 1]]))
    >>> df1.get_values()
    array([[0, 1, 2],
           [1, 1, 1],
           [2, 3, 0]])

    >>> df1.get_variables()
    ['A', 'B', 'C']

    >>> df2.get_values()
    array([[0, 0, 0],
           [0, 1, 1],
           [0, 1, 1],
           [1, 2, 3],
           [0, 2, 1]])

    >>> df1.get_num_of_samples()
    3

    >>> print(df1)
       A    B    C
     ---  ---  ---
       0    1    2
       1    1    1
       2    3    0

    >>> df1==df2
    False

    >>> df1!=df2
    True

    """
    def __init__(self, variables, values=None):
        """
        Initializes a DataFrame Class.

        Parameters
        ----------
        variables: list, array-like
        List of variables whose data is given in the sample.

        values: list, array-like
        List of values in the sample.
        """
        if isinstance(variables, six.string_types):
            raise TypeError("Variables: Expected type list or array like, got string")
        self.variables = list(variables)

        self.values = np.empty([0, len(variables)], dtype=int)
        if values is not None:
            self.add_samples(values)

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
        return self.values.copy()

    def get_num_of_samples(self):
        """
        Returns the number of samples in the DataFrame

        Returns
        -------
        integer: number of samples

        Examples
        --------
        >>> from pgmpy.factors import DataFrame
        >>> df = DataFrame(['A', 'B', 'C'], [1,1,1,0,0,1,1,1,0,1,0,1])
        >>> df.get_num_of_samples()
        4
        """
        return self.values.shape[0]

    def add_samples(self, values):
        """
        Adds samples in the DataFrame.

        Parameters
        ----------
        values: list, array-like
        List of values in the samples.

        Examples
        --------
        >>> from pgmpy.factors import DataFrame
        >>> df = DataFrame(['A', 'B', 'C'], [1,1,1,0,0,1,1,1,0,1,0,1])
        >>> df.get_values()
        array([[1, 1, 1],
               [0, 0, 1],
               [1, 1, 0],
               [1, 0, 1]])
        >>> df.add_samples([1,1,0,2,2,2])
        >>> df.get_values()
        array([[1, 1, 1],
               [0, 0, 1],
               [1, 1, 0],
               [1, 0, 1],
               [1, 1, 0],
               [2, 2, 2]])
        """
        values = np.asarray(values)

        if values.dtype != int:
            raise TypeError("Values: Expected type int, got ", values.dtype)

        if values.size % len(self.get_variables()):
            raise ValueError("Values: Number of samples for each variable should be same.")

        values = values.reshape(-1, len(self.get_variables()))
        self.values = np.concatenate((self.values, values))

    def __str__(self):
        return(tabulate(self.get_values(), headers=self.get_variables()))

    def __eq__(self, other):
        if not(isinstance(other, DataFrame)):
            return False

        if set(other.get_variables()) != set(self.get_variables()):
            return False

        # re-arrange other_values columns according to self.values columns
        other_values = other.get_values()[:, list(other.get_variables().index(var)
                                          for var in self.get_variables())]
        other_values.sort(axis=0)
        self_values = self.get_values()
        self_values.sort(axis=0)
        if not np.array_equal(other_values, self_values):
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)
