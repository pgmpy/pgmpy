import numpy as np

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
        # For P(Y| X1, X2) = N(0.2x1 + 0.3x2 + 1.0; 1)

        >>> cpd = FunctionalCPD(
            variable="x3",
            fn=lambda parent_sample: np.random.normal(
                0.2 * parent_sample["x1"] + 0.3 * parent_sample["x2"] + 1.0, 1
            ),
            parents=["x1", "x2"],
        )
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

    def sample(self, parent_sample=None):
        """
        Simulates a value for the variable based on its CPD.

        Parameters:
        ----------
        parent_sample: dict, optional
            A dictionary of parent variable names and their sampled values.
            Default is None, which is valid if the variable has no parents.

        Returns:
        -------
        sampled_value: float
            The sampled value for the variable.
        """
        parent_sample = parent_sample or {}
        if not all(parent in parent_sample for parent in self.parents):
            missing_parents = [p for p in self.parents if p not in parent_sample]
            raise ValueError(f"Missing values for parent variables: {missing_parents}")
        return self.fn(parent_sample)

    def __repr__(self):
        if self.parents:
            parents_str = ", ".join(self.parents)
            return (
                f"FunctionalCPD(variable={self.variable}, "
                f"parents=[{parents_str}], "
                f"function={self.fn.__name__ if hasattr(self.fn, '__name__') else 'custom'})"
            )
        else:
            return (
                f"FunctionalCPD(variable={self.variable}, "
                f"function={self.fn.__name__ if hasattr(self.fn, '__name__') else 'custom'})"
            )
