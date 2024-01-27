#!/usr/bin/env python3
from __future__ import annotations
from numbers import Number
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pgmpy.factors.discrete import DiscreteFactor


class FactorDict(dict):
    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, marginals: List[Tuple[str, ...]]
    ) -> FactorDict:
        if df.isnull().values.any():
            raise ValueError("df cannot contain None or np.NaN values.")

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
