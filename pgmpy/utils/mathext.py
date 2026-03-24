from collections import namedtuple
from itertools import chain, combinations
from typing import Any

import numpy as np

from pgmpy import config, logger
from pgmpy.utils import compat_fns

State = namedtuple("State", ["var", "state"])


def cartesian(arrays: list[Any], out: np.ndarray | None = None) -> np.ndarray:
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
    shape = [len(x) for x in arrays]
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    result: np.ndarray
    if out is None:
        result = np.empty_like(ix, dtype=dtype)
    else:
        result = out

    for n, _ in enumerate(arrays):
        result[:, n] = arrays[n][ix[:, n]]

    return result


def _adjusted_weights(weights: np.ndarray):
    """
    Adjusts the weights such that it sums to 1. When the total weights is less
    than or greater than 1 by 1e-3, add/substracts the difference from the last
    element of weights. If the difference is greater than 1e-3, throws an error.

    Parameters
    ----------
    weights: 1-D numpy array
        The array for which to do the adjustment.

    Example
    -------
    >>> a = np.array([0.1111111] * 9)
    >>> result = _adjusted_weights(a)
    >>> bool(np.isclose(result.sum(), 1.0))
    True
    >>> int(np.sum(result != 0.1111111))  # Exactly one element was adjusted
    1
    """
    error = 1 - compat_fns.sum(weights)
    error_value = float(compat_fns.to_numpy(error))
    if abs(error_value) > 1e-3:
        raise ValueError("The probability values do not sum to 1.")
    elif error_value != 0:
        logger.warning(f"Probability values don't exactly sum to 1. Differ by: {error_value}. Adjusting values.")
        weights[compat_fns.argmax(weights)] += error_value

    return weights


def sample_discrete(values, weights: np.ndarray | list[np.ndarray], size=1, seed: int | None = None):
    """
    Generate a sample of given size, given a probability mass function.

    Parameters
    ----------
    values: numpy.array
        Array of all possible values that the random variable can take.

    weights: numpy.array or list of numpy.array
        Array(s) representing the PMF of the random variable.

    size: int
        Size of the sample to be generated.

    seed: int (default: None)
        If a value is provided, sets the seed for numpy.random.

    Returns
    -------
    samples: numpy.array
        Array of values of the random variable sampled from the given PMF.

    Example
    -------
    >>> import numpy as np
    >>> from pgmpy.utils.mathext import sample_discrete
    >>> values = np.array(["v_0", "v_1", "v_2"])
    >>> probabilities = np.array([0.2, 0.5, 0.3])
    >>> sample_discrete(values, probabilities, 10, seed=0).tolist()
    ['v_1', 'v_2', 'v_1', 'v_1', 'v_1', 'v_1', 'v_1', 'v_2', 'v_2', 'v_1']
    """
    if config.get_backend() == "numpy":
        if seed is not None:
            np.random.seed(seed)
        weights = compat_fns.to_numpy(weights)
        if weights.ndim == 1:
            return np.random.choice(compat_fns.to_numpy(values), size=size, p=_adjusted_weights(weights))
        else:
            samples = np.zeros(size, dtype=int)
            unique_weights, counts = np.unique(weights, axis=0, return_counts=True)
            for index, size in enumerate(counts):
                samples[(weights == unique_weights[index]).all(axis=1)] = np.random.choice(
                    compat_fns.to_numpy(values),
                    size=size,
                    p=_adjusted_weights(unique_weights[index]),
                )
            return samples
    else:
        import torch

        if seed is not None:
            torch.manual_seed(seed)

        values = np.asarray(list(values) if isinstance(values, range) else values)
        weights = torch.as_tensor(weights, dtype=config.get_dtype(), device=config.get_device())
        size = int(size)

        if weights.ndim == 1:
            sampled_indices = torch.multinomial(_adjusted_weights(weights.clone()), num_samples=size, replacement=True)
        else:
            sampled_indices = torch.empty(size, dtype=torch.long, device=config.get_device())
            unique_weights, inverse, counts = torch.unique(weights, dim=0, return_inverse=True, return_counts=True)
            for index, count in enumerate(counts.tolist()):
                sampled_indices[inverse == index] = torch.multinomial(
                    _adjusted_weights(unique_weights[index].clone()),
                    num_samples=count,
                    replacement=True,
                )

        return values[compat_fns.to_numpy(sampled_indices).astype(int)]


def sample_discrete_maps(
    states: np.ndarray,
    weight_indices: np.ndarray,
    index_to_weight: np.ndarray,
    size=1,
    seed: int | None = None,
):
    """
    Generate a sample of given size, given a probability mass function.

    Parameters
    ----------
    states: numpy.array
        Array of all possible states that the random variable can take.

    weight_indices: numpy.array
        Array with the weight indices for each sample

    index_to_weight: numpy.array
        Array mapping each weight index to a specific weight

    size: int
        Size of the sample to be generated.

    seed: int (default: None)
        If a value is provided, sets the seed for numpy.random.

    Returns
    -------
    samples: numpy.array
        Array of values of the random variable sampled from the given PMF.

    Example
    -------
    >>> import numpy as np
    >>> from pgmpy.utils.mathext import sample_discrete
    >>> values = np.array(["v_0", "v_1", "v_2"])
    >>> probabilities = np.array([0.2, 0.5, 0.3])
    >>> sample_discrete(values, probabilities, 10, seed=0).tolist()
    ['v_1', 'v_2', 'v_1', 'v_1', 'v_1', 'v_1', 'v_1', 'v_2', 'v_2', 'v_1']
    """
    if config.get_backend() == "numpy":
        if seed is not None:
            np.random.seed(seed)

        states = np.array(states)
        weight_indices = compat_fns.to_numpy(weight_indices)
        index_to_weight = {key: compat_fns.to_numpy(value) for key, value in index_to_weight.items()}
        size = int(size)

        samples = np.zeros(size, dtype=int)
        unique_weight_indices, counts = np.unique(weight_indices, return_counts=True)

        for weight_size, weight_index in zip(counts, unique_weight_indices):
            samples[(weight_indices == weight_index)] = np.random.choice(
                states, size=weight_size, p=_adjusted_weights(index_to_weight[weight_index])
            )
        return samples
    else:
        import torch

        if seed is not None:
            torch.manual_seed(seed)

        states = np.asarray(list(states) if isinstance(states, range) else states)
        weight_indices = torch.as_tensor(weight_indices, dtype=torch.long, device=config.get_device())
        index_to_weight = {
            key: torch.as_tensor(value, dtype=config.get_dtype(), device=config.get_device())
            for key, value in index_to_weight.items()
        }
        size = int(size)

        samples = torch.empty(size, dtype=torch.long, device=config.get_device())
        unique_weight_indices, counts = torch.unique(weight_indices, return_counts=True)

        for weight_index, weight_size in zip(unique_weight_indices.tolist(), counts.tolist()):
            samples[weight_indices == weight_index] = torch.multinomial(
                _adjusted_weights(index_to_weight[weight_index].clone()),
                num_samples=weight_size,
                replacement=True,
            )

        return states[compat_fns.to_numpy(samples).astype(int)]


def powerset(l_input: list):
    """
    Generates all subsets of list `l_input` (as tuples).

    Example
    -------
    >>> from pgmpy.utils.mathext import powerset
    >>> list(powerset([1, 2, 3]))
    [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    return chain.from_iterable(combinations(l_input, r) for r in range(len(l_input) + 1))
