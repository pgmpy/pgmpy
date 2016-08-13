from collections import namedtuple

import numpy as np
from itertools import combinations, chain
from pgmpy.extern.six.moves import map


State = namedtuple('State', ['var', 'state'])


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def sample_discrete(values, weights, size=1):
    """
    Generate a sample of given size, given a probability mass function.

    Parameters
    ----------
    values: numpy.array: Array of all possible values that the random variable
            can take.
    weights: numpy.array or list of numpy.array: Array(s) representing the PMF of the random variable.
    size: int: Size of the sample to be generated.

    Returns
    -------
    numpy.array: of values of the random variable sampled from the given PMF.

    Example
    -------
    >>> import numpy as np
    >>> from pgmpy.utils.mathext import sample_discrete
    >>> values = np.array(['v_0', 'v_1', 'v_2'])
    >>> probabilities = np.array([0.2, 0.5, 0.3])
    >>> sample_discrete(values, probabilities, 10)
    array(['v_1', 'v_1', 'v_0', 'v_1', 'v_2', 'v_0', 'v_1', 'v_1', 'v_1',
      'v_2'], dtype='<U3')
    """
    weights = np.array(weights)
    if weights.ndim == 1:
        return np.random.choice(values, size=size, p=weights)
    else:
        return np.fromiter(map(lambda t: np.random.choice(values, p=t), weights), dtype='int')


def powerset(l):
    """
    Generates all subsets of list `l` (as tuples).

    Example
    -------
    >>> from pgmpy.utils.mathext import powerset
    >>> list(powerset([1,2,3]))
    [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
    """
    return chain.from_iterable(combinations(l, r) for r in range(len(l)+1))
