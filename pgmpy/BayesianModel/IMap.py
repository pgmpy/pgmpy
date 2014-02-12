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