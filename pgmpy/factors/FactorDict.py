#!/usr/bin/env python3
from __future__ import annotations

from numbers import Number

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from pgmpy.factors.base import factor_product
from pgmpy.factors.discrete import DiscreteFactor


class FactorDict(dict):
    @classmethod
    def from_dataframe(cls, df, marginals):
        """Create a `FactorDict` from a given set of marginals.

        Parameters
        ----------
        df: pandas DataFrame object

        marginals: List[Tuple[str]]
            List of Tuples containing the names of the marginals.

        Returns
        -------
        Factor dictionary: FactorDict
            FactorDict with each marginal's Factor representing the empirical
                frequency of the marginal from the dataset.
        """
        if df.isnull().values.any():
            raise ValueError("df cannot contain None or np.nan values.")

        factor_dict = cls({})
        for marginal in marginals:
            # Subset of columns arranged in a lexographical ordering.
            _df = df.loc[:, list(marginal)].sort_values(list(marginal))
            cardinality = list(_df.nunique())
            # Since we have sorted the columns, this encoding will
            # also be sorted lexographically.
            encoded = OrdinalEncoder().fit_transform(_df)
            factor_dict[marginal] = DiscreteFactor(
                variables=marginal,
                cardinality=cardinality,
                values=np.histogramdd(sample=encoded, bins=cardinality)[0].flatten(),
                state_names={
                    column: sorted(_df[column].unique().tolist()) for column in marginal
                },
            )
        return factor_dict

    def get_factors(self):
        return set(self.values())

    def __mul__(self, const):
        return FactorDict({clique: const * self[clique] for clique in self})

    def __rmul__(self, const):
        return self.__mul__(const)

    def __add__(self, other):
        return FactorDict(
            {clique: self[clique] + other for clique in self}
            if isinstance(other, Number)
            else {clique: self[clique] + other[clique] for clique in self}
        )

    def __sub__(self, other):
        return self + -1 * other

    def dot(self, other):
        return sum((self[clique] * other[clique]).values.sum() for clique in self)

    def product(self):
        return factor_product(*self.get_factors())
