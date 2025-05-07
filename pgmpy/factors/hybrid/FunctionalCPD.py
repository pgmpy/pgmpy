import numpy as np
import pandas as pd
import pyro

from pgmpy.factors.base import BaseFactor


class FunctionalCPD(BaseFactor):
    """
    Defines a Functional CPD.

    Functional CPD can represent any arbitrary conditional probability
    distribution where the distribution to represented is defined by function
    (input as parameter) which calls pyro.sample function.

    Parameters
    ----------
    variable: str
        Name of the variable for which this CPD is defined.

    fn: callable
        A lambda function that takes a dictionary of parent variable values
        and returns a sampled value for the variable by calling pyro.sample.

    parents: list[str], optional
        List of parent variable names (default is None for no parents).

    Examples
    --------
    # For P(X3| X1, X2) = N(0.2x1 + 0.3x2 + 1.0; 1), we can write

    >>> from pgmpy.factors.hybrid import FunctionalCPD
    >>> import pyro.distributions as dist
    >>> cpd = FunctionalCPD(
    ...    variable="x3",
    ...    fn=lambda parent_sample: dist.Normal(
    ...        0.2 * parent_sample["x1"] + 0.3 * parent_sample["x2"] + 1.0, 1),
    ...    parents=["x1", "x2"])
    >>> cpd.variable
    'x3'
    >>> cpd.parents
    ['x1', 'x2']
    """

    def __init__(self, variable, fn, parents=[]):
        self.variable = variable
        if not callable(fn):
            raise ValueError("`fn` must be a callable function.")
        self.fn = fn
        self.parents = parents if parents else []
        self.variables = [variable] + self.parents

    def sample(self, n_samples=100, parent_sample=None):
        """
        Simulates a value for the variable based on its CPD.

        Parameters
        ----------

        n_samples: int, (default: 100)
            The number of samples to generate.

        parent_sample: pandas.DataFrame, optional
            A DataFrame where each column represents a parent variable and rows are samples.

        Returns
        -------
        sampled_values: numpy.ndarray
            Array of sampled values for the variable.

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> import pyro.distributions as dist
        >>> cpd = FunctionalCPD(
        ...    variable="x3",
        ...    fn=lambda parent_sample: dist.Normal(
        ...        1.0 + 0.2 * parent_sample["x1"] + 0.3 * parent_sample["x2"], 1),
        ...    parents=["x1", "x2"])

        >>> parent_samples = pd.DataFrame({'x1' : [5, 10], 'x2' : [1, -1]})
        >>> cpd.sample(2, parent_samples)

        """
        sampled_values = []

        if parent_sample is not None:
            if not isinstance(parent_sample, pd.DataFrame):
                raise TypeError("`parent_sample` must be a pandas DataFrame.")

            if not all(parent in parent_sample.columns for parent in self.parents):
                missing_parents = [
                    p for p in self.parents if p not in parent_sample.columns
                ]
                raise ValueError(
                    f"Missing values for parent variables: {missing_parents}"
                )
            if len(parent_sample) != n_samples:
                raise ValueError("Length of `parent_sample` must match `n_samples`.")

            for i in range(n_samples):
                sampled_values.append(
                    pyro.sample(
                        f"{self.variable}", self.fn(parent_sample.iloc[i, :])
                    ).item()
                )
        else:
            for i in range(n_samples):
                sampled_values.append(
                    pyro.sample(f"{self.variable}", self.fn(parent_sample)).item()
                )

        sampled_values = np.array(sampled_values)

        return sampled_values

    def __str__(self):
        fn_name = "lambda fun." if self.fn.__name__ == "<lambda>" else self.fn.__name__
        if self.parents:
            return f"P({self.variable} | {', '.join(self.parents)}) = {fn_name}"
        return f"P({self.variable}) = {fn_name}"

    def __repr__(self):
        return f"<FunctionalCPD: {self.__str__()}> at {hex(id(self))}"
