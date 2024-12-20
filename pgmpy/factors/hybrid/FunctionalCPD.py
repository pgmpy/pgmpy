import numpy as np
import pandas as pd

from pgmpy.factors.base import BaseFactor


class FunctionalCPD(BaseFactor):
    def __init__(self, variable, fn, parents=None):
        """

        Parameters:
        ----------

        variable: str
            Name of the variable for which this CPD is defined.

        fn: callable
            A function that takes a dictionary of parent variable values
            and returns a sampled value for the variable.

        parents: list[str], optional
            List of parent variable names (default is None for no parents).

        Examples
        --------
        # For P(X3| X1, X2) = N(0.2x1 + 0.3x2 + 1.0; 1), we can write

        >>> from pgmpy.factors.hybrid import LinearGaussianCPD
        >>> cpd = FunctionalCPD(
        ...    variable="x3",
        ...    fn=lambda parent_sample: np.random.normal(
        ...        0.2 * parent_sample["x1"] + 0.3 * parent_sample["x2"] + 1.0, 1),
        ...    parents=["x1", "x2"])

        >>> cpd.variable
        'x3'

        >>> cpd.parents
        ['x1', 'x2']
        """
        self.variable = variable
        if not callable(fn):
            raise ValueError("`fn` must be a callable function.")
        self.fn = fn
        self.parents = parents if parents else []
        self.variables = [variable] + self.parents

    def sample(self, n_samples=100, parent_sample=None):
        """
        Simulates a value for the variable based on its CPD.

        Parameters:
        ----------

        n_samples: int, (default: 100)
            The number of samples to generate.

        parent_sample: pandas.DataFrame, optional
            A DataFrame where each column represents a parent variable and rows are samples.

        Returns:
        -------
        sampled_values: numpy.ndarray
            Array of sampled values for the variable.

        """
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

            sampled_values = self.fn(parent_sample)
        else:
            sampled_values = []
            for _ in range(n_samples):
                sampled_values.append(self.fn(parent_sample))

            sampled_values = np.array(sampled_values)

        return sampled_values

    def __str__(self):
        fn_name = "f(mean, std)" if self.fn.__name__ == "<lambda>" else self.fn.__name__
        if self.parents:
            return f"P({self.variable} | {', '.join(self.parents)}) = {fn_name}"
        return f"P({self.variable}) = {fn_name}"

    def __repr__(self):
        return f"<FunctionalCPD: {self.__str__()}> at {hex(id(self))}"


# cpd = FunctionalCPD(
#     variable="x3",
#     fn=lambda parent_sample: np.random.normal(
#         0.2 * parent_sample["x1"] + 0.3 * parent_sample["x2"] + 1.0, 1),
#     parents=["x1", "x2"]
# )
# print(cpd)

# import pandas as pd
# # Create a DataFrame for parent samples
# parent_df = pd.DataFrame({
#     "x1": np.random.normal(0, 1, 100),
#     "x2": np.random.normal(0, 1, 100)
# })

# # Sample values for x3
# samples = cpd.sample(n_samples=100, parent_sample=parent_df)
# print(samples)

# cpd = FunctionalCPD(
#     variable="x1",
#     fn=lambda _: np.random.normal(1, 1))

# print(cpd.sample())
