#!/usr/bin/env python
from pgmpy.estimators import BaseEstimator

class StructureScore(BaseEstimator):
    def __init__(self, data, **kwargs):
        """
        Base class for structure scoring classes in pgmpy. Scoring classes are
        used to measure how well a model is able to describe the given data set.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        """
        super(StructureScore, self).__init__(data, **kwargs)

    def score(self, model):
        pass

    def local_score(self, variable, parents):
        pass
