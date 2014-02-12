class IMap:
    """
    Base class for IMap.
    IMap is a set of Conditional Independence (eg: "X is independent of Y given Z" where X, Y and Z
    are random variables) or Independence(eg: "X is independent of Y" where X and Y
    are random variables) assertions.

    Public Methods
    --------------
    add_assertions
    get_imap
    """
    def __init__(self, *assertions):
        """
        Initialize the IMap with Conditional Independence or Independence assertions.

        Parameters
        ----------
        assertions: Lists or Tuples
                Each assertion is a list or tuple of the form: [variable, independent_of and given]
                eg: assertion ['X', 'Y', 'Z'] would be X is independent of Y given Z.

        Examples
        --------
        Creating an IMap object with one independence assertion: Random Variable X is independent of Y
        >>> imap = IMap(['X', 'Y'])

        Creating an IMap object with three conditional independence assertions:
        First assertion is Random Variable X is independent of Y given Z.
        >>> imap = IMap(['X', 'Y', 'Z'],
        >>>             ['a', ['b', 'c'], 'd'],
        >>>             ['l', ['m', 'n'], 'o'])
        """
        self.imap = set()
        for assertion in assertions:
            self.imap.add(IndependenceAssertion(assertion[0], assertion[1], assertion[2]))

    def get_imap(self):
        """
        Returns the imap which is a set of IndependenceAssertion objects.
        """
        return self.imap

    def add_assertions(self, *assertions):
        """
        Adds assertions to imap.

        Parameters
        ----------
        assertions: Lists or Tuples
                Each assertion is a list or tuple of variable, independent_of and given.

        Examples
        --------
        >>> imap = IMap()
        >>> imap.add_assertions(['X', 'Y', 'Z'])
        >>> imap.add_assertions(['a', ['b', 'c'], 'd'])
        """
        for assertion in assertions:
            self.imap.add(IndependenceAssertion(assertion[0], assertion[1], assertion[2]))


class IndependenceAssertion:
    """
    Represents Independence Assertion.

    Has 3 attributes: variable, independent_of, given.
    The attributes for (U |  X, Y |  Z) read as
                         ---
    Random Variable U is independent of X and Y given Z would be:

    variable = {U}
    independent_of = {X, Y}
    given = {Z}

    Public Methods
    --------------
    get_assertion
    set_assertion
    """
    def __init__(self, variable=[], independent_of=[], given=[]):
        """
        Initialize an IndependenceAssertion object with variable, independent_of and given attributes.

        Parameters
        ----------
        variable: String or List
                Random Variable which is independent.

        independent_of: String or list of strings.
                Random Variables from which variable is independent

        given: String or list of strings.
                Random Variables given which variable is independent of independent_of.

        Examples
        --------
        >>> assertion = IndependenceAssertion('U', 'X')
        >>> assertion = IndependenceAssertion('U', ['X', 'Y'])
        >>> assertion = IndependenceAssertion('U', ['X', 'Y'], 'Z')
        >>> assertion = IndependenceAssertion('U', ['X', 'Y'], ['Z', 'A'])
        """
        if variable and not independent_of:
            raise Exception.Needed('independent_of')
        if any([independent_of, given]) and not variable:
            raise Exception.Needed('variable')
        if given and not all([variable, independent_of]):
            raise Exception.Needed('variable' if not variable else 'independent_of')

        self.variable = set(self._return_list_if_str(variable))
        self.independent_of = set(self._return_list_if_str(independent_of))
        self.given = set(self._return_list_if_str(given))

    @staticmethod
    def _return_list_if_str(variable):
        """
        If variable is a string returns a list containing variable.
        Else returns variable itself.
        """
        if isinstance(variable, str):
            return [variable]
        else:
            return variable

    def get_assertion(self):
        """
        Returns a tuple of the attributes: variable, independent_of, given.

        See Also
        --------
        set_assertion
        """
        return self.variable, self.independent_of, self.given

    def set_assertion(self, variable, independent_of, given=[]):
        """
        Sets the attributes variable, independent_of and given.

        Parameters
        ----------
        variable: String or List
                Random Variable which is independent.

        independent_of: String or list of strings.
                Random Variables with which variable is independent of

        given: String or list of strings.
                Random Variables with which variable is independent of independent_of given given.

        Example
        -------
        For a random variable U independent of X and Y given Z, the function should be called as
        >>> set_assertion('U', ['X', 'Y'], 'Z')
        >>> set_assertion('U', ['X', 'Y'], ['Z', 'A'])

        See Also
        --------
        get_assertion
        """
        self.__init__(variable, independent_of, given)