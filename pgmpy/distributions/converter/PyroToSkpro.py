from skpro.distributions.normal import Normal as SkproNormal


def _convert_normal(samples, index, columns, name="obs"):
    """
    Convert the posterior samples obtained from MCMC and SVI using Pyro
    into the Normal distribution format provided by skpro.
    """
    y_samples = samples[name]
    mean = y_samples.mean(dim=0).detach().reshape(-1, 1).cpu().numpy()
    sigma = y_samples.std(dim=0, unbiased=False).detach().reshape(-1, 1).cpu().numpy()

    return SkproNormal(
        mu=mean,
        sigma=sigma,
        index=index,
        columns=columns,
    )


class PyroToSkpro:
    """
    Convert the posterior samples obtained from MCMC and SVI using Pyro into the distribution format provided by skpro.

    `add()` adds a new conversion method,
    while `set()` specifies which conversion method to use when `convert()` is called.
    `remove()` deletes a specific conversion method, and `get()` returns the list of available conversion methods.


    Examples
    --------
    Create an default `PyroToSkpro`

    >>> from pgmpy.distributions.converter.PyroToSkpro import PyroToSkpro
    >>> import torch
    >>> import pandas as pd
    >>> import numpy as np

    >>> samples = {
    ...     "obs": torch.tensor(
    ...         [
    ...             [1.0, 2.0],
    ...             [3.0, 4.0],
    ...             [5.0, 6.0],
    ...         ]
    ...     )
    ... }
    >>> converter = PyroToSkpro("normal")
    >>> normal = converter.convert(samples, pd.RangeIndex(0,2), ["normal"], "obs")
    >>> normal # doctest: +SKIP
    Normal()

    The currently implemented converter classes can be viewed using `get()`.

    >>> converter.get()
    ['normal']

    You can add a custom converter, retrieve the distribution produced by that converter,
    and manage it using the `add()`, `set()`, and `remove()` methods.
    The example below demonstrates how to add and use a custom converter that produces a Laplace distribution.
    >>> from skpro.distributions.laplace import Laplace

    >>> def _convert_laplace(samples, index, columns, name="obs"):
    ...     y_samples = samples[name]
    ...     mean = y_samples.mean(dim=0).detach().reshape(-1, 1).cpu().numpy()
    ...     scale = y_samples.std(dim=0, unbiased=False).detach().reshape(-1, 1).cpu().numpy()
    ...     return Laplace(
    ...         mu=mean,
    ...         scale=scale,
    ...         index=index,
    ...         columns=columns,
    ...     )
    >>> converter = PyroToSkpro()
    >>> converter.add("laplace", _convert_laplace) # doctest: +SKIP
    >>> converter.remove("normal") # doctest: +SKIP
    >>> converter.get() # doctest: +SKIP
    ['laplace']
    >>> converter.set("laplace") # doctest: +SKIP
    >>> laplace = converter.convert(samples, pd.RangeIndex(0,2), ["laplace"], "obs") # doctest: +SKIP
    >>> laplace # doctest: +SKIP
    Laplace()

    """

    def __init__(self, posterior_type: str = "normal"):
        _DEFAULT_CONVERTERS = {
            "normal": _convert_normal,
        }

        self.posterior_type = posterior_type
        self._converters = _DEFAULT_CONVERTERS

    def convert(self, samples, index, columns, name="obs"):
        """
        Converts Pyro posterior samples into the skpro distribution format.
        """
        converter = self._converters.get(self.posterior_type)

        if converter is None:
            raise ValueError(f"{self.posterior_type} is not ...")

        return converter(samples, index, columns, name=name)

    def add(
        self,
        posterior_type,
        converter_fn,
    ):
        """
        Adds a new conversion method.
        """
        if not callable(converter_fn):
            raise TypeError

        self._converters[posterior_type] = converter_fn
        return self

    def set(self, posterior_type):
        """
        Specifies which conversion method to use when `convert()` is called.
        """
        self.posterior_type = posterior_type
        return self

    def remove(
        self,
        posterior_type,
    ):
        """
        Removes a specific conversion method.
        """
        self._converters.pop(posterior_type)
        return self

    def get(self):
        """
        Returns a list of all available conversion methods.
        """
        return list(self._converters.keys())
