# -*- coding: utf-8 -*-

import itertools


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
    closure
    entails
    is_equivalent
    """

    def __init__(self, *assertions):
        self.independencies = []
        self.add_assertions(*assertions)

    def __str__(self):
        string = "\n".join([str(assertion) for assertion in self.independencies])
        return string

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, Independencies):
            return False
        return all(
            independency in other.get_assertions()
            for independency in self.get_assertions()
        ) and all(
            independency in self.get_assertions()
            for independency in other.get_assertions()
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def contains(self, assertion):
        """
        Returns `True` if `assertion` is contained in this `Independencies`-object,
        otherwise `False`.

        Parameters
        ----------
        assertion: IndependenceAssertion()-object

        Examples
        --------
        >>> from pgmpy.independencies import Independencies, IndependenceAssertion
        >>> ind = Independencies(['A', 'B', ['C', 'D']])
        >>> IndependenceAssertion('A', 'B', ['C', 'D']) in ind
        True
        >>> # does not depend on variable order:
        >>> IndependenceAssertion('B', 'A', ['D', 'C']) in ind
        True
        >>> # but does not check entailment:
        >>> IndependenceAssertion('X', 'Y', 'Z') in Independencies(['X', 'Y'])
        False
        """
        if not isinstance(assertion, IndependenceAssertion):
            raise TypeError(
                "' in <Independencies()>' requires IndependenceAssertion"
                + " as left operand, not {0}".format(type(assertion))
            )

        return assertion in self.get_assertions()

    __contains__ = contains

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
                    self.independencies.append(
                        IndependenceAssertion(assertion[0], assertion[1], assertion[2])
                    )
                except IndexError:
                    self.independencies.append(
                        IndependenceAssertion(assertion[0], assertion[1])
                    )

    def closure(self):
        """
        Returns a new `Independencies()`-object that additionally contains those `IndependenceAssertions`
        that are implied by the the current independencies (using with the `semi-graphoid axioms
        <https://en.wikipedia.org/w/index.php?title=Conditional_independence&oldid=708760689#Rules_of_conditional_independence>`_;
        see (Pearl, 1989, `Conditional Independence and its representations
        <http://www.cs.technion.ac.il/~dang/journal_papers/pearl1989conditional.pdf>`_)).

        Might be very slow if more than six variables are involved.

        Examples
        --------
        >>> from pgmpy.independencies import Independencies
        >>> ind1 = Independencies(('A', ['B', 'C'], 'D'))
        >>> ind1.closure()
        (A _|_ B | D, C)
        (A _|_ B, C | D)
        (A _|_ B | D)
        (A _|_ C | D, B)
        (A _|_ C | D)

        >>> ind2 = Independencies(('W', ['X', 'Y', 'Z']))
        >>> ind2.closure()
        (W _|_ Y)
        (W _|_ Y | X)
        (W _|_ Z | Y)
        (W _|_ Z, X, Y)
        (W _|_ Z)
        (W _|_ Z, X)
        (W _|_ X, Y)
        (W _|_ Z | X)
        (W _|_ Z, Y | X)
        [..]
        """

        def single_var(var):
            "Checks if var represents a single variable"
            if not hasattr(var, "__iter__"):
                return True
            else:
                return len(var) == 1

        def sg0(ind):
            "Symmetry rule: 'X ⟂ Y | Z' -> 'Y ⟂ X | Z'"
            return IndependenceAssertion(ind.event2, ind.event1, ind.event3)

        # since X⟂Y|Z == Y⟂X|Z in pgmpy, sg0 (symmetry) is not used as an axiom/rule.
        # instead we use a decorator for the other axioms to apply them on both sides
        def apply_left_and_right(func):
            def symmetric_func(*args):
                if len(args) == 1:
                    return func(args[0]) + func(sg0(args[0]))
                if len(args) == 2:
                    return (
                        func(*args)
                        + func(args[0], sg0(args[1]))
                        + func(sg0(args[0]), args[1])
                        + func(sg0(args[0]), sg0(args[1]))
                    )

            return symmetric_func

        @apply_left_and_right
        def sg1(ind):
            "Decomposition rule: 'X ⟂ Y,W | Z' -> 'X ⟂ Y | Z', 'X ⟂ W | Z'"
            if single_var(ind.event2):
                return []
            else:
                return [
                    IndependenceAssertion(ind.event1, ind.event2 - {elem}, ind.event3)
                    for elem in ind.event2
                ]

        @apply_left_and_right
        def sg2(ind):
            "Weak Union rule: 'X ⟂ Y,W | Z' -> 'X ⟂ Y | W,Z', 'X ⟂ W | Y,Z' "
            if single_var(ind.event2):
                return []
            else:
                return [
                    IndependenceAssertion(
                        ind.event1, ind.event2 - {elem}, {elem} | ind.event3
                    )
                    for elem in ind.event2
                ]

        @apply_left_and_right
        def sg3(ind1, ind2):
            "Contraction rule: 'X ⟂ W | Y,Z' & 'X ⟂ Y | Z' -> 'X ⟂ W,Y | Z'"
            if ind1.event1 != ind2.event1:
                return []

            Y = ind2.event2
            Z = ind2.event3
            Y_Z = ind1.event3
            if Y < Y_Z and Z < Y_Z and Y.isdisjoint(Z):
                return [IndependenceAssertion(ind1.event1, ind1.event2 | Y, Z)]
            else:
                return []

        # apply semi-graphoid axioms as long as new independencies are found.
        all_independencies = set()
        new_inds = set(self.independencies)

        while new_inds:
            new_pairs = (
                set(itertools.permutations(new_inds, 2))
                | set(itertools.product(new_inds, all_independencies))
                | set(itertools.product(all_independencies, new_inds))
            )

            all_independencies |= new_inds
            new_inds = set(
                sum(
                    [sg1(ind) for ind in new_inds]
                    + [sg2(ind) for ind in new_inds]
                    + [sg3(*inds) for inds in new_pairs],
                    [],
                )
            )
            new_inds -= all_independencies

        return Independencies(*list(all_independencies))

    def entails(self, entailed_independencies):
        """
        Returns `True` if the `entailed_independencies` are implied by this `Independencies`-object, otherwise `False`.
        Entailment is checked using the semi-graphoid axioms.

        Might be very slow if more than six variables are involved.

        Parameters
        ----------
        entailed_independencies: Independencies()-object

        Examples
        --------
        >>> from pgmpy.independencies import Independencies
        >>> ind1 = Independencies([['A', 'B'], ['C', 'D'], 'E'])
        >>> ind2 = Independencies(['A', 'C', 'E'])
        >>> ind1.entails(ind2)
        True
        >>> ind2.entails(ind1)
        False
        """
        if not isinstance(entailed_independencies, Independencies):
            return False

        implications = self.closure().get_assertions()
        return all(
            ind in implications for ind in entailed_independencies.get_assertions()
        )

    def is_equivalent(self, other):
        """
        Returns True if the two Independencies-objects are equivalent, otherwise False.
        (i.e. any Bayesian Network that satisfies the one set
        of conditional independencies also satisfies the other).

        Might be very slow if more than six variables are involved.

        Parameters
        ----------
        other: Independencies()-object

        Examples
        --------
        >>> from pgmpy.independencies import Independencies
        >>> ind1 = Independencies(['X', ['Y', 'W'], 'Z'])
        >>> ind2 = Independencies(['X', 'Y', 'Z'], ['X', 'W', 'Z'])
        >>> ind3 = Independencies(['X', 'Y', 'Z'], ['X', 'W', 'Z'], ['X', 'Y', ['W','Z']])
        >>> ind1.is_equivalent(ind2)
        False
        >>> ind1.is_equivalent(ind3)
        True
        """
        return self.entails(other) and other.entails(self)

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
    r"""
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
    """

    def __init__(self, event1=[], event2=[], event3=[]):
        r"""
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
            raise ValueError("event2 needs to be specified")
        if any([event2, event3]) and not event1:
            raise ValueError("event1 needs to be specified")
        if event3 and not all([event1, event2]):
            raise ValueError(
                "event1" if not event1 else "event2" + " needs to be specified"
            )

        self.event1 = frozenset(self._return_list_if_str(event1))
        self.event2 = frozenset(self._return_list_if_str(event2))
        self.event3 = frozenset(self._return_list_if_str(event3))

    def __str__(self):
        if self.event3:
            return "({event1} _|_ {event2} | {event3})".format(
                event1=", ".join(self.event1),
                event2=", ".join(self.event2),
                event3=", ".join(self.event3),
            )
        else:
            return "({event1} _|_ {event2})".format(
                event1=", ".join(self.event1), event2=", ".join(self.event2)
            )

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, IndependenceAssertion):
            return False
        return (self.event1, self.event2, self.event3) == other.get_assertion() or (
            self.event2,
            self.event1,
            self.event3,
        ) == other.get_assertion()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((frozenset((self.event1, self.event2)), self.event3))

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

        Examples
        --------
        >>> from pgmpy.independencies import IndependenceAssertion
        >>> asser = IndependenceAssertion('X', 'Y', 'Z')
        >>> asser.get_assertion()
        """
        return self.event1, self.event2, self.event3

    def latex_string(self):
        return r"%s \perp %s \mid %s" % (
            ", ".join(self.event1),
            ", ".join(self.event2),
            ", ".join(self.event3),
        )
