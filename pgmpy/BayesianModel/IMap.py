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
    get_factorized_product
    """
    def __init__(self, *assertions):
        """
        Initialize the IMap with Conditional Independence or Independence assertions.

        Parameters
        ----------
        assertions: Lists or Tuples
                Each assertion is a list or tuple of the form: [event1, event2 and event3]
                eg: assertion ['X', 'Y', 'Z'] would be X is independent of Y given Z.

        Examples
        --------
        Creating an IMap object with one independence assertion: Random Variable X is independent of Y
        >>> imap = IMap(['X', 'Y'])

        Creating an IMap object with three conditional independence assertions:
        First assertion is Random Variable X is independent of Y given Z.
        >>> imap = IMap(['X', 'Y', 'Z'],
        ...             ['a', ['b', 'c'], 'd'],
        ...             ['l', ['m', 'n'], 'o'])
        """
        self.imap = set()
        for assertion in assertions:
            self.imap.add(IndependenceAssertion(assertion[0], assertion[1], assertion[2]))

    def get_imap(self):
        """
        Returns the imap which is a set of IndependenceAssertion objects.

        See Also
        --------
        add_assertions

        Examples
        --------
        >>> imap = IMap(['X', 'Y', 'Z'])
        >>> imap.get_imap()
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

    def get_factorized_product(self, random_variables=None, latex=False):
        #TODO: Write this whole function
        """
        The problem right now is that the factorized product for all
        P(A, B, C), P(B, A, C) etc should be same but on solving normally
        we get different results which have to be simplified to a simpler
        form. How to do that ??? and also how to decide which is the most
        simplified form???
        """
        pass


class IndependenceAssertion:
    """
    Represents Conditional Independence or Independence assertion.

    Each assertion has 3 attributes: event1, event2, event3.
    The attributes for (U || X, Y | Z) read as
                         ---
    Random Variable U is independent of X and Y given Z would be:

    event1 = {U}
    event2 = {X, Y}
    event3 = {Z}

    Public Methods
    --------------
    get_assertion
    set_assertion
    """
    def __init__(self, event1=[], event2=[], event3=[]):
        """
        Initialize an IndependenceAssertion object with event1, event2 and event3 attributes.

        Parameters
        ----------

                event2
                  ^
      event1     /   event3
         ^      /     ^
         |     /      |
        (U || X, Y | Z) read as Random variable U is independent of X and Y given Z.
          ---

        event1: String or List of strings
                Random Variable which is independent.

        event2: String or list of strings.
                Random Variables from which event1 is independent

        event3: String or list of strings.
                Random Variables given which event1 is independent of event2.

        Examples
        --------
        >>> assertion = IndependenceAssertion('U', 'X')
        >>> assertion = IndependenceAssertion('U', ['X', 'Y'])
        >>> assertion = IndependenceAssertion('U', ['X', 'Y'], 'Z')
        >>> assertion = IndependenceAssertion(['U', 'V'], ['X', 'Y'], ['Z', 'A'])
        """
        if event1 and not event2:
            raise Exception.Needed('event2 needed')
        if any([event2, event3]) and not event1:
            raise Exception.Needed('variable')
        if event3 and not all([event1, event2]):
            raise Exception.Needed('variable' if not variable else 'independent_of')

        self.event1 = set(self._return_list_if_str(event1))
        self.event2 = set(self._return_list_if_str(event2))
        self.event3 = set(self._return_list_if_str(event3))

    @staticmethod
    def _return_list_if_str(event):
        """
        If variable is a string returns a list containing variable.
        Else returns variable itself.
        """
        if isinstance(event, str):
            return [event]
        else:
            return event

    def get_assertion(self):
        """
        Returns a tuple of the attributes: variable, independent_of, given.

        See Also
        --------
        set_assertion

        Examples
        --------
        >>> asser = IndependenceAssertion('X', 'Y', 'Z')
        >>> asser.get_assertion()
        """
        return self.event1, self.event2, self.event3

    def set_assertion(self, event1, event2, event3=[]):
        """
        Sets the attributes event1, event2 and event3.

        Parameters
        ----------

               event2
                  ^
       event1    /   event3
         ^      /     ^
         |     /      |
        (U || X, Y | Z) read as Random variable U is independent of X and Y given Z.
          ---

        event1: String or List
                Random Variable which is independent.

        event2: String or list of strings.
                Random Variables from which event1 is independent

        event3: String or list of strings.
                Random Variables given which event1 is independent of event2.

        See Also
        --------
        get_assertion

        Example
        -------
        For a random variable U independent of X and Y given Z, the function should be called as
        >>> asser = IndependenceAssertion()
        >>> asser.set_assertion('U', ['X', 'Y'], 'Z')
        >>> asser.set_assertion('U', ['X', 'Y'], ['Z', 'A'])
        """
        self.__init__(event1, event2, event3)