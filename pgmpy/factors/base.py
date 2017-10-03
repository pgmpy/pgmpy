from abc import abstractmethod

from pgmpy.extern.six.moves import reduce


class BaseFactor(object):
    """
    Base class for Factors. Any Factor implementation should inherit this class.
    """
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def is_valid_cpd(self):
        pass


def factor_product(*args):
    """
    Returns factor product over `args`.

    Parameters
    ----------
    args: `BaseFactor` instances.
        factors to be multiplied

    Returns
    -------
    BaseFactor: `BaseFactor` representing factor product over all the `BaseFactor` instances in args.

    Examples
    --------
    >>> from pgmpy.factors.discrete import DiscreteFactor
    >>> from pgmpy.factors import factor_product
    >>> phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
    >>> phi2 = DiscreteFactor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
    >>> phi = factor_product(phi1, phi2)
    >>> phi.variables
    ['x1', 'x2', 'x3', 'x4']
    >>> phi.cardinality
    array([2, 3, 2, 2])
    >>> phi.values
    array([[[[ 0,  0],
             [ 4,  6]],

            [[ 0,  4],
             [12, 18]],

            [[ 0,  8],
             [20, 30]]],


           [[[ 6, 18],
             [35, 49]],

            [[ 8, 24],
             [45, 63]],

            [[10, 30],
             [55, 77]]]])
    """
    if not all(isinstance(phi, BaseFactor) for phi in args):
        raise TypeError("Arguments must be factors")
    # Check if all of the arguments are of the same type
    elif len(set(map(type, args))) != 1:
            raise NotImplementedError("All the args are expected to ",
                                      "be instances of the same factor class.")

    return reduce(lambda phi1, phi2: phi1 * phi2, args)


def factor_divide(phi1, phi2):
    """
    Returns `DiscreteFactor` representing `phi1 / phi2`.

    Parameters
    ----------
    phi1: Factor
        The Dividend.

    phi2: Factor
        The Divisor.

    Returns
    -------
    DiscreteFactor: `DiscreteFactor` representing factor division `phi1 / phi2`.

    Examples
    --------
    >>> from pgmpy.factors.discrete import DiscreteFactor
    >>> from pgmpy.factors import factor_product
    >>> phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
    >>> phi2 = DiscreteFactor(['x3', 'x1'], [2, 2], range(1, 5))
    >>> phi = factor_divide(phi1, phi2)
    >>> phi.variables
    ['x1', 'x2', 'x3']
    >>> phi.cardinality
    array([2, 3, 2])
    >>> phi.values
    array([[[ 0.        ,  0.33333333],
            [ 2.        ,  1.        ],
            [ 4.        ,  1.66666667]],

           [[ 3.        ,  1.75      ],
            [ 4.        ,  2.25      ],
            [ 5.        ,  2.75      ]]])
    """
    if not isinstance(phi1, BaseFactor) or not isinstance(phi2, BaseFactor):
        raise TypeError("phi1 and phi2 should be factors instances")

    # Check if all of the arguments are of the same type
    elif type(phi1) != type(phi2):
        raise NotImplementedError("All the args are expected to be instances",
                                  "of the same factor class.")

    return phi1.divide(phi2, inplace=False)
