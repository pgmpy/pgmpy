import pandas as pd
from skbase.base import BaseObject

from pgmpy.utils import preprocess_data


class BaseStructureScore(BaseObject):
    """
    Abstract base class for structure scoring in pgmpy.

    Structure scores are used to evaluate how well a given Bayesian network structure
    fits observed data. This class should not be used directly. Use one of the derived
    classes such as `K2`, `BDeu`, `BIC`, or `AIC` for concrete scoring methods.

    Structure scores are central to model selection in Bayesian networks and are
    particularly useful when comparing candidate network structures in discrete data scenarios.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame in which each column represents a variable. Missing values should
        be marked as `numpy.nan`. Note: Columns with `numpy.nan` will have dtype `float`.

    state_names : dict, optional
        Dictionary mapping each variable name to the set of its discrete states.
        If not specified, the observed values in the data are used as possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.estimators import K2
    >>> # Create random data sample with 3 variables, where B and C are identical:
    >>> data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 2)), columns=list("AB"))
    >>> data["C"] = data["B"]
    >>> model1 = DiscreteBayesianNetwork([["A", "B"], ["A", "C"]])
    >>> model2 = DiscreteBayesianNetwork([["A", "B"], ["B", "C"]])
    >>> K2(data).score(model1)
    -24242.367348745247
    >>> K2(data).score(model2)
    -16273.793897051042

    Notes
    -----
    - Use this class as a base for implementing custom structure scores.
    - Use derived classes (`K2`, `BDeu`, `BIC`, `AIC`) for standard scoring approaches.
    - If you provide data with continuous variables or incompatible states, a `ValueError` may be raised.
    - For best results, ensure all variables are discrete and states are correctly specified.

    Raises
    ------
    ValueError
        If data contains unsupported (non-discrete) types, or if the variables
        in the model do not match the data columns.

    References
    ----------
    Koller & Friedman, Probabilistic Graphical Models: Principles and Techniques, 2009, Section 18.3.
    """

    def __init__(self, data, state_names=None, **kwargs):
        # if data is None:
        #    self.data = None
        #    self.dtypes = None
        self.data, self.dtypes = preprocess_data(data)

        if self.data is not None:
            self.variables = list(data.columns.values)

            if not isinstance(state_names, dict):
                self.state_names = {
                    var: self._collect_state_names(var) for var in self.variables
                }
            else:
                self.state_names = dict()
                for var in self.variables:
                    if var in state_names:
                        if not set(self._collect_state_names(var)) <= set(
                            state_names[var]
                        ):
                            raise ValueError(
                                f"Data contains unexpected states for variable: {var}."
                            )
                        self.state_names[var] = state_names[var]
                    else:
                        self.state_names[var] = self._collect_state_names(var)

    def _collect_state_names(self, variable):
        "Return a list of states that the variable takes in the data."
        states = sorted(list(self.data.loc[:, variable].dropna().unique()))
        return states

    def score(self, model):
        """
        Computes a structure score for a given Bayesian network model.

        This method evaluates how well the specified `DiscreteBayesianNetwork`
        fits the observed data, using the structure score metric implemented in the subclass.
        The higher (or less negative) the score, the better the fit between the model and the data.

        Parameters
        ----------
        model : DiscreteBayesianNetwork
            The Bayesian network whose structure is to be scored. All nodes in the
            model must correspond to columns in the input data.

        Returns
        -------
        score : float
            The computed structure score representing the model's fit to the data.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.estimators import K2
        >>> # create random data sample with 3 variables, where B and C are identical:
        >>> data = pd.DataFrame(
        ...     np.random.randint(0, 5, size=(5000, 2)), columns=list("AB")
        ... )
        >>> data["C"] = data["B"]
        >>> K2(data).score(DiscreteBayesianNetwork([["A", "B"], ["A", "C"]]))
        -24242.367348745247
        >>> K2(data).score(DiscreteBayesianNetwork([["A", "B"], ["B", "C"]]))
        -16273.793897051042

        Raises
        ------
        ValueError
            If the model contains nodes not present in the data columns, or if the
            data contains unsupported variable types.
        """

        score = 0
        for node in model.nodes():
            score += self.local_score(node, list(model.predecessors(node)))
        score += self.structure_prior(model)
        return score

    def structure_prior(self, model):
        """
        Computes the (log) prior distribution over Bayesian network structures.

        This method returns a uniform prior by default and is currently unused in scoring.
        Override this method in subclasses to implement custom prior distributions
        over network structures.

        Parameters
        ----------
        model : DiscreteBayesianNetwork
            The Bayesian network model for which the structure prior is to be computed.

        Returns
        -------
        prior : float
            The log prior probability of the given model structure. By default, returns 0.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.estimators import K2
        >>> model = DiscreteBayesianNetwork([("A", "B")])
        >>> score = K2(data)
        >>> prior = score.structure_prior(model)
        >>> print(prior)
        0
        """
        return 0

    def structure_prior_ratio(self, operation):
        """
        Computes the log ratio of prior probabilities for a proposed change to the model structure.

        This method returns the log prior probability ratio for a structural operation
        (e.g., adding, removing, or reversing an edge) in the Bayesian network. By default,
        it assumes a uniform prior and returns 0, meaning no structural operation is favored.

        Parameters
        ----------
        operation : tuple or object
            The proposed operation on the Directed Acyclic Graph (DAG), typically represented as a tuple
            describing the change (such as ('add', 'A', 'B') for adding an edge from A to B).

        Returns
        -------
        prior_ratio : float
            The log ratio of the prior probabilities for the proposed operation. By default, returns 0.

        Examples
        --------
        >>> from pgmpy.estimators import K2
        >>> op = ("add", "A", "B")  # Example operation
        >>> score = K2(data)
        >>> ratio = score.structure_prior_ratio(op)
        >>> print(ratio)
        0
        """
        return 0

    def state_counts(
        self,
        variable,
        parents=[],
        weighted=False,
        reindex=True,
    ):
        """
        Return counts how often each state of 'variable' occurred in the data.
        If a list of parents is provided, counting is done conditionally
        for each state configuration of the parents.

        Parameters
        ----------
        variable: string
            Name of the variable for which the state count is to be done.

        parents: list
            Optional list of variable parents, if conditional counting is desired.
            Order of parents in list is reflected in the returned DataFrame

        weighted: bool
            If True, data must have a `_weight` column specifying the weight of the
            datapoint (row). If False, each datapoint has a weight of `1`.

        reindex: bool
            If True, returns a data frame with all possible parents state combinations
            as the columns. If False, drops the state combinations which are not
            present in the data.

        Returns
        -------
        state_counts: pandas.DataFrame
            Table with state counts for 'variable'

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.estimators import BaseEstimator
        >>> data = pd.DataFrame(
        ...     data={
        ...         "A": ["a1", "a1", "a2"],
        ...         "B": ["b1", "b2", "b1"],
        ...         "C": ["c1", "c1", "c2"],
        ...     }
        ... )
        >>> estimator = BaseEstimator(data)
        >>> estimator.state_counts(variable="A").values
        array([[2],
               [1]])
        >>> estimator.state_counts(variable="C", parents=["A", "B"]).values
        array([[1., 1., 0., 0.],
               [0., 0., 1., 0.]])
        """
        parents = list(parents)

        if weighted and ("_weight" not in self.data.columns):
            raise ValueError("data must contain a `_weight` column if weighted=True")

        if not parents:
            # count how often each state of 'variable' occurred
            if weighted:
                state_count_data = self.data.groupby([variable], observed=True)[
                    "_weight"
                ].sum()
            else:
                state_count_data = self.data.loc[:, variable].value_counts()

            state_counts = (
                state_count_data.reindex(self.state_names[variable])
                .fillna(0)
                .to_frame()
            )

        else:
            parents_states = [self.state_names[parent] for parent in parents]
            # count how often each state of 'variable' occurred, conditional on parents' states
            if weighted:
                state_count_data = (
                    self.data.groupby([variable] + parents, observed=True)["_weight"]
                    .sum()
                    .unstack(parents)
                )

            else:
                state_count_data = (
                    self.data.groupby([variable] + parents, observed=True)
                    .size()
                    .unstack(parents)
                )

            if not isinstance(state_count_data.columns, pd.MultiIndex):
                state_count_data.columns = pd.MultiIndex.from_arrays(
                    [state_count_data.columns]
                )

            if reindex:
                # reindex rows & columns to sort them and to add missing ones
                # missing row    = some state of 'variable' did not occur in data
                # missing column = some state configuration of current 'variable's parents
                #                  did not occur in data
                row_index = self.state_names[variable]
                column_index = pd.MultiIndex.from_product(parents_states, names=parents)
                state_counts = state_count_data.reindex(
                    index=row_index, columns=column_index
                ).fillna(0)
            else:
                state_counts = state_count_data.fillna(0)

        return state_counts
