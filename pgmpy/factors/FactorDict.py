#!/usr/bin/env python3
from __future__ import annotations

from numbers import Number

import numpy as np
import pandas as pd

from pgmpy.factors.base import factor_product
from pgmpy.factors.discrete import DiscreteFactor


class FactorDict(dict):
    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, marginals: list[tuple[str]]
    ) -> FactorDict:
        """Create a `FactorDict` from a given set of marginals.

        Parameters
        ----------
        df: pandas DataFrame object
            Input data frame containing the variables.

        marginals: List[Tuple[str]]
            List of Tuples containing the names of the marginals.

        Returns
        -------
        FactorDict
            Dictionary with each marginal's Factor representing the empirical
            frequency of the marginal from the dataset.
        """
        if df.isnull().values.any():
            raise ValueError("df cannot contain None or np.nan values.")

        factor_dict = cls({})

        for marginal in marginals:
            # Subset of columns arranged in lexicographical order
            _df = df.loc[:, list(marginal)].sort_values(list(marginal))
            cardinality = list(_df.nunique())

            # Custom ordinal encoding implementation
            encoded = np.zeros(_df.shape, dtype=int)
            state_names = {}

            for i, col in enumerate(marginal):
                unique_vals = sorted(_df[col].unique())
                state_names[col] = unique_vals
                val_to_code = {val: code for code, val in enumerate(unique_vals)}
                encoded[:, i] = _df[col].map(val_to_code).values

            # Compute histogram (empirical frequencies)
            hist, _ = np.histogramdd(sample=encoded, bins=cardinality)

            factor_dict[marginal] = DiscreteFactor(
                variables=marginal,
                cardinality=cardinality,
                values=hist.flatten(),
                state_names=state_names,
            )

        return factor_dict

    def get_factors(self) -> set[DiscreteFactor]:
        """Get all factors in the dictionary."""
        return set(self.values())

    def __mul__(self, const: Number) -> FactorDict:
        """Multiply all factors by a constant."""
        if not isinstance(const, Number):
            raise TypeError("Can only multiply by numbers")
        return FactorDict({clique: const * self[clique] for clique in self})

    def __rmul__(self, const: Number) -> FactorDict:
        """Right multiplication by a constant."""
        return self.__mul__(const)

    def __add__(self, other: Number | FactorDict) -> FactorDict:
        """Add a constant or another FactorDict to this one."""
        if isinstance(other, Number):
            return FactorDict({clique: self[clique] + other for clique in self})
        elif isinstance(other, FactorDict):
            return FactorDict({clique: self[clique] + other[clique] for clique in self})
        else:
            raise TypeError("Can only add numbers or FactorDict instances")

    def __sub__(self, other: Number | FactorDict) -> FactorDict:
        """Subtract a constant or another FactorDict from this one."""
        return self.__add__(-1 * other)

    def dot(self, other: FactorDict) -> float:
        """Compute the dot product with another FactorDict."""
        if not isinstance(other, FactorDict):
            raise TypeError("Dot product only defined between FactorDict instances")
        return sum((self[clique] * other[clique]).values.sum() for clique in self)

    def product(self) -> DiscreteFactor:
        """Compute the product of all factors in the dictionary."""
        return factor_product(*self.get_factors())
