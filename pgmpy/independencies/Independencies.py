from pgmpy import exceptions
from pgmpy.extern import six


class Independencies(object):
    """
    Base class for independencies.
    independencies class represents a set of Conditional Independence
    assertions (eg: "X is independent of Y given Z" where X, Y and Z
    are random variables) or Independence assertions (eg: "X is
    independent of Y" where X and Y are random variables).
    Initialize the independencies Class with Conditional Independence
    assertions or Independence assertions.

    Parameters
    ----------
    assertions: Lists or Tuples
            Each assertion is a list or tuple of the form: [event1,
            event2 and event3]
            eg: assertion ['X', 'Y', 'Z'] would be X is independent
            of Y given Z.

    Examples
    --------
    Creating an independencies object with one independence assertion:
    Random Variable X is independent of Y

    >>> independencies = independencies(['X', 'Y'])

    Creating an independencies object with three conditional
    independence assertions:
    First assertion is Random Variable X is independent of Y given Z.

    >>> independencies = independencies(['X', 'Y', 'Z'],
    ...             ['a', ['b', 'c'], 'd'],
    ...             ['l', ['m', 'n'], 'o'])

    Public Methods
    --------------
    add_assertions
    get_assertions
    get_factorized_product
    """
    def __init__(self, *assertions):
        self.independencies = []
        self.add_assertions(*assertions)

    def __str__(self):
        string = '\n'.join([str(assertion) for assertion in self.independencies])
        return string

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, Independencies):
            return False
        other_assertions = other.get_assertions()
        return all(self_independency in other_assertions for self_independency in self.get_assertions())

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_assertions(self):
        """
        Returns the independencies object which is a set of IndependenceAssertion objects.

        Examples
        --------
        >>> from pgmpy.independencies import Independencies
        >>> independencies = Independencies(['X', 'Y', 'Z'])
        >>> independencies.get_assertions()
        """
        return self.independencies

    def add_assertions(self, *assertions):
        """
        Adds assertions to independencies.

        Parameters
        ----------
        assertions: Lists or Tuples
                Each assertion is a list or tuple of variable, independent_of and given.

        Examples
        --------
        >>> from pgmpy.independencies import Independencies
        >>> independencies = Independencies()
        >>> independencies.add_assertions(['X', 'Y', 'Z'])
        >>> independencies.add_assertions(['a', ['b', 'c'], 'd'])
        """
        for assertion in assertions:
            if isinstance(assertion, IndependenceAssertion):
                self.independencies.append(assertion)
            else:
                try:
                    self.independencies.append(IndependenceAssertion(assertion[0], assertion[1], assertion[2]))
                except IndexError:
                    self.independencies.append(IndependenceAssertion(assertion[0], assertion[1]))

        # TODO: write reduce function.
    def reduce(self):
        """
        Add function to remove duplicate Independence Assertions
        """
        pass

    def latex_string(self):
        """
        Returns a list of string.
        Each string represents the IndependenceAssertion in latex.
        """
        return [assertion.latex_string() for assertion in self.get_assertions()]

    def get_factorized_product(self, random_variables=None, latex=False):
        # TODO: Write this whole function
        #
        # The problem right now is that the factorized product for all
        # P(A, B, C), P(B, A, C) etc should be same but on solving normally
        # we get different results which have to be simplified to a simpler
        # form. How to do that ??? and also how to decide which is the most
        # simplified form???
        #
        pass


class IndependenceAssertion(object):
    """
    Represents Conditional Independence or Independence assertion.

    Each assertion has 3 attributes: event1, event2, event3.
    The attributes for

    .. math:: U \perp X, Y | Z

    is read as: Random Variable U is independent of X and Y given Z would be:

    event1 = {U}

    event2 = {X, Y}

    event3 = {Z}

    Parameters
    ----------
    event1: String or List of strings
            Random Variable which is independent.

    event2: String or list of strings.
            Random Variables from which event1 is independent

    event3: String or list of strings.
            Random Variables given which event1 is independent of event2.

    Examples
    --------
    >>> from pgmpy.independencies import IndependenceAssertion
    >>> assertion = IndependenceAssertion('U', 'X')
    >>> assertion = IndependenceAssertion('U', ['X', 'Y'])
    >>> assertion = IndependenceAssertion('U', ['X', 'Y'], 'Z')
    >>> assertion = IndependenceAssertion(['U', 'V'], ['X', 'Y'], ['Z', 'A'])


    Public Methods
    --------------
    get_assertion
    set_assertion
    """
    def __init__(self, event1=[], event2=[], event3=[]):
        """
        Initialize an IndependenceAssertion object with event1, event2 and event3 attributes.

                  event2
                  ^
      event1     /   event3
         ^      /     ^
         |     /      |
        (U || X, Y | Z) read as Random variable U is independent of X and Y given Z.
          ---
        """
        if event1 and not event2:
            raise exceptions.RequiredError('event2 needed')
        if any([event2, event3]) and not event1:
            raise exceptions.RequiredError('event1')
        if event3 and not all([event1, event2]):
            raise exceptions.RequiredError('event1' if not event1 else 'event2')

        self.event1 = set(self._return_list_if_str(event1))
        self.event2 = set(self._return_list_if_str(event2))
        self.event3 = set(self._return_list_if_str(event3))

    def __str__(self):
        if self.event3:
            return('({event1} _|_ {event2} | {event3})'.format(event1=', '.join(self.event1),
                                                               event2=', '.join(self.event2),
                                                               event3=', '.join(self.event3)))
        else:
            return('({event1} _|_ {event2})'.format(event1=', '.join(self.event1),
                                                    event2=', '.join(self.event2)))

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, IndependenceAssertion):
            return False
        self_assertions = self.get_assertion()
        other_assertions = other.get_assertion()
        if len(self_assertions) != len(other_assertions):
            return False
        self_assertions = set(six.moves.map(frozenset, self_assertions))
        other_assertions = set(six.moves.map(frozenset, other_assertions))
        if self_assertions == other_assertions:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def _return_list_if_str(event):
        """
        If variable is a string returns a list containing variable.
        Else returns variable itself.
        """
        if isinstance(event, six.string_types):
            return [event]
        else:
            return event

    def get_assertion(self):
        """
        Returns a tuple of the attributes: variable, independent_of, given.

        Examples
        --------
        >>> from pgmpy.independencies import IndependenceAssertion
        >>> asser = IndependenceAssertion('X', 'Y', 'Z')
        >>> asser.get_assertion()
        """
        return self.event1, self.event2, self.event3

    def set_assertion(self, event1, event2, event3=[]):
        """
        Sets the attributes event1, event2 and event3.

        .. math:: U \perp X, Y | Z

        event1 = {U}

        event2 = {X, Y}

        event3 = {Z}

        Parameters
        ----------
        event1: String or List
                Random Variable which is independent.

        event2: String or list of strings.
                Random Variables from which event1 is independent

        event3: String or list of strings.
                Random Variables given which event1 is independent of event2.

        Example
        -------
        For a random variable U independent of X and Y given Z, the function should be called as
        >>> from pgmpy.independencies import IndependenceAssertion
        >>> asser = IndependenceAssertion()
        >>> asser.set_assertion('U', ['X', 'Y'], 'Z')
        >>> asser.set_assertion('U', ['X', 'Y'], ['Z', 'A'])
        """
        self.__init__(event1, event2, event3)

    def latex_string(self):
        return ('%s \perp %s \mid %s' % (', '.join(self.event1), ', '.join(self.event2),
                                         ', '.join(self.event3)))
